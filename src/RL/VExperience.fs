module VExperience

open System
open MathNet.Numerics
open FSharpx.Collections
open FSharp.Collections.ParallelSeq
open TorchSharp
open TorchSharp.Fun 

type VExperience = {State:float32[]; NextState:float32[]; Action:int; Reward:float32; Done:bool; Priority:float32}
type VExperienceBufferUniform = {Buffer:RandomAccessList<VExperience>; Max:int}
type VExperienceBufferStratified = {BufferMap:Map<int,RandomAccessList<VExperience>>; Max:int; MinSamplesPerStrata:int}
type VExperienceBufferPrioritized = {Buffer:RandomAccessList<VExperience>; Max:int; Alpha:float; PriorityEps:float}
type VExperienceBuffer =
    | UniformSampled of VExperienceBufferUniform
    | StratifiedSampled of VExperienceBufferStratified
    | PrioritizedSampled of VExperienceBufferPrioritized
    with member this.Length() = 
            match this with
            | UniformSampled e -> e.Buffer.Length
            | StratifiedSampled m -> m.BufferMap |> Map.toSeq |> Seq.sumBy (fun (_,v) -> v.Length)
            | PrioritizedSampled p -> p.Buffer.Length

let createUniformSampled maxExperiance = UniformSampled {Buffer=RandomAccessList.empty; Max=maxExperiance}

let createStratifiedSampled maxExperincePerStrata minSamplesPerStrata =    
        StratifiedSampled {BufferMap=Map.empty; Max=maxExperincePerStrata; MinSamplesPerStrata=minSamplesPerStrata}

let createPrioritizedSampled maxExperiance alpha priorityEps =
        PrioritizedSampled {Buffer=RandomAccessList.empty; Max=maxExperiance; Alpha=alpha; PriorityEps=priorityEps}
        
let private appendExperience max exp (ls:RandomAccessList<VExperience>) = 
    let ls = RandomAccessList.cons exp ls
    if ls.Length > int (float max * 1.3)  then
        //trim list
        ls |> RandomAccessList.toSeq |> Seq.take max |> RandomAccessList.ofSeq
    else
        ls

let append exp = function
    | UniformSampled buff -> UniformSampled {buff with Buffer = appendExperience buff.Max exp buff.Buffer}
    | StratifiedSampled buff -> 
        let expBuff = buff.BufferMap |> Map.tryFind exp.Action |> Option.defaultValue RandomAccessList.empty
        let expBuff = appendExperience buff.Max exp expBuff
        StratifiedSampled {buff with BufferMap = buff.BufferMap |> Map.add exp.Action expBuff}
    | PrioritizedSampled buff ->
        let maxPriority =
            if buff.Buffer.IsEmpty then
                1.0f
            else
                buff.Buffer |> Seq.maxBy _.Priority |> _.Priority
        let exp' = {exp with Priority = maxPriority}
        PrioritizedSampled {buff with Buffer = appendExperience buff.Max exp' buff.Buffer}

let private sampleExperience max n (expBuff:RandomAccessList<VExperience>) =
    if expBuff.Length <= n then 
        Seq.toArray expBuff 
    else 
        let maxLen = min max  expBuff.Length //temporarily buffer may be longer than max
        let idx = torch.randperm(int64 maxLen,dtype=torch.int) |> Tensor.getData<int> 
        [|for i in 0..n-1 -> expBuff.[idx.[i]]|]

let private failIfEmptyBuffer = function
    | UniformSampled buff when buff.Buffer.IsEmpty ->
        failwith "cannot operate on an empty replay buffer"
    | StratifiedSampled buff when buff.BufferMap |> Map.forall (fun _ xs -> xs.IsEmpty) ->
        failwith "cannot operate on an empty replay buffer"
    | PrioritizedSampled buff when buff.Buffer.IsEmpty ->
        failwith "cannot operate on an empty replay buffer"
    | _ -> ()

let private stratifiedSample n (buff:VExperienceBufferStratified) =
    if n <= 0 then
        [||]
    else
        let strata =
            buff.BufferMap
            |> Map.toArray
            |> Array.choose (fun (k, exps) ->
                let available = min buff.Max exps.Length
                if available > 0 then Some (k, exps, available) else None)
        if Array.isEmpty strata then
            failwith "cannot sample from an empty stratified replay buffer"
        else
            let sampleTarget = min n (strata |> Array.sumBy (fun (_, _, available) -> available))
            let minPerStrata =
                if buff.MinSamplesPerStrata <= 0 then
                    0
                else
                    buff.MinSamplesPerStrata
            let availableCounts = strata |> Array.map (fun (_, _, available) -> available)
            let totalAvailable = availableCounts |> Array.sum |> float
            let proportional =
                availableCounts
                |> Array.map (fun available -> float sampleTarget * float available / totalAvailable)
            let allocations = Array.zeroCreate strata.Length
            let mutable remaining = sampleTarget

            if minPerStrata > 0 then
                let mutable keepGoing = true
                while remaining > 0 && keepGoing do
                    keepGoing <- false
                    for i in 0 .. allocations.Length - 1 do
                        let targetMinimum = min minPerStrata availableCounts.[i]
                        if remaining > 0 && allocations.[i] < targetMinimum then
                            allocations.[i] <- allocations.[i] + 1
                            remaining <- remaining - 1
                            keepGoing <- true

            while remaining > 0 do
                let nextChoice =
                    proportional
                    |> Array.mapi (fun i desired ->
                        let capacity = availableCounts.[i] - allocations.[i]
                        let score = desired - float allocations.[i]
                        i, capacity, score)
                    |> Array.filter (fun (_, capacity, _) -> capacity > 0)
                    |> Array.sortByDescending (fun (_, _, score) -> score)
                    |> Array.tryHead
                match nextChoice with
                | Some (i, _, _) ->
                    allocations.[i] <- allocations.[i] + 1
                    remaining <- remaining - 1
                | None ->
                    remaining <- 0

            Array.zip strata allocations
            |> Array.collect (fun ((_, exps, _), count) -> sampleExperience buff.Max count exps)
            |> Array.randomShuffle

let sample n = function 
    | UniformSampled buff -> sampleExperience buff.Max n buff.Buffer
    | StratifiedSampled buff -> stratifiedSample n buff
    | PrioritizedSampled buff ->
        let sampled, _, _ = 
            let beta = 0.0
            let available = min buff.Max buff.Buffer.Length
            let exps = buff.Buffer |> Seq.truncate available |> Seq.toArray
            if n <= 0 || Array.isEmpty exps then [||], [||], [||] else
                let priorities =
                    exps
                    |> Array.map (fun exp -> max (float32 buff.PriorityEps) exp.Priority |> float |> fun p -> p ** buff.Alpha)
                let totalPriority = priorities |> Array.sum
                let probabilities =
                    if totalPriority <= 0.0 then
                        Array.init exps.Length (fun _ -> 1.0 / float exps.Length)
                    else
                        priorities |> Array.map (fun p -> p / totalPriority)
                let sampleCount = min n exps.Length
                let cdf = probabilities |> Array.scan (+) 0.0 |> Array.tail
                let sampledIdxs =
                    Array.init sampleCount (fun _ ->
                        let r = torch.rand([|1L|]).ToDouble()
                        let mutable idx = Array.BinarySearch(cdf, r)
                        if idx < 0 then idx <- ~~~idx
                        min idx (cdf.Length - 1))
                let sampled = sampledIdxs |> Array.map (fun idx -> exps.[idx])
                let weights =
                    sampledIdxs
                    |> Array.map (fun idx ->
                        let p = max 1e-12 probabilities.[idx]
                        ((float exps.Length * p) ** (-beta)) |> float32)
                sampled, sampledIdxs, weights
        sampled

let private prioritizedSample n beta (buff:VExperienceBufferPrioritized) =
    if n <= 0 then
        [||], [||], [||]
    else
        let available = min buff.Max buff.Buffer.Length
        let exps = buff.Buffer |> Seq.truncate available |> Seq.toArray
        if Array.isEmpty exps then
            failwith "cannot sample from an empty prioritized replay buffer"
        else
            let priorities =
                exps
                |> Array.map (fun exp -> max (float32 buff.PriorityEps) exp.Priority |> float |> fun p -> p ** buff.Alpha)
            let totalPriority = priorities |> Array.sum
            let probabilities =
                if totalPriority <= 0.0 then
                    Array.init exps.Length (fun _ -> 1.0 / float exps.Length)
                else
                    priorities |> Array.map (fun p -> p / totalPriority)
            let sampleCount = min n exps.Length
            let cdf = probabilities |> Array.scan (+) 0.0 |> Array.tail
            let sampledIdxs =
                Array.init sampleCount (fun _ ->
                    let r = torch.rand([|1L|]).ToDouble()
                    let mutable idx = Array.BinarySearch(cdf, r)
                    if idx < 0 then idx <- ~~~idx
                    min idx (cdf.Length - 1))
            let sampled = sampledIdxs |> Array.map (fun idx -> exps.[idx])
            let weights =
                sampledIdxs
                |> Array.map (fun idx ->
                    let p = max 1e-12 probabilities.[idx]
                    ((float exps.Length * p) ** (-beta)) |> float32)
            let maxWeight =
                weights
                |> Array.map float
                |> Array.max
                |> max 1e-12
                |> float32
            let normalizedWeights = weights |> Array.map (fun w -> w / maxWeight)
            sampled, sampledIdxs, normalizedWeights

let recall n buff =
    failIfEmptyBuffer buff
    let exps = sample n buff
    if Array.isEmpty exps then failwith "cannot recall an empty minibatch from replay buffer"
    let states     = exps |> Array.map (fun x->x.State)
    let nextStates = exps |> Array.map (fun x->x.NextState)
    let actions    = exps |> Array.map (fun x->x.Action)
    let rewards    = exps |> Array.map (fun x -> x.Reward)
    let dones      = exps |> Array.map (fun x->x.Done)
    states,nextStates,rewards,actions,dones

let recallPrioritized n beta = function
    | PrioritizedSampled buff ->
        failIfEmptyBuffer (PrioritizedSampled buff)
        let exps, indices, weights = prioritizedSample n beta buff
        if Array.isEmpty exps then failwith "cannot recall an empty minibatch from prioritized replay buffer"
        let states = exps |> Array.map _.State
        let nextStates = exps |> Array.map _.NextState
        let actions = exps |> Array.map _.Action
        let rewards = exps |> Array.map _.Reward
        let dones = exps |> Array.map _.Done
        states,nextStates,rewards,actions,dones,indices,weights
    | _ ->
        failwith "prioritized recall called for a non-prioritized replay buffer"

let updatePriorities indices priorities = function
    | PrioritizedSampled buff ->
        let priorityMap =
            Array.zip indices priorities
            |> Array.groupBy fst
            |> Array.map (fun (idx, xs) -> idx, xs |> Array.averageBy snd)
            |> Map.ofArray

        let updated =
            buff.Buffer
            |> Seq.toArray
            |> Array.mapi (fun i exp ->
                match priorityMap |> Map.tryFind i with
                | Some priority ->
                    let nextPriority = max (float32 buff.PriorityEps) priority
                    {exp with Priority = nextPriority}
                | None ->
                    exp)
            |> RandomAccessList.ofSeq

        PrioritizedSampled {buff with Buffer = updated}
    | other ->
        other

//use built-in F# types for serialization - works better for .fsx scripts
type VTser = int * int option * int * string * float option * float option * Map<int,List<float32[]*float32[]*int*float32*bool*float32>> 
        // max * minSamples * state length * replayMode * alpha * eps * buffer payload

let private exportExperience (expBuff:RandomAccessList<VExperience>)  = 
    expBuff
    |> Seq.map (fun x-> 
            x.State,
            x.NextState,
            x.Action,
            x.Reward,
            x.Done,
            x.Priority)
    |> Seq.toList

let private importExperience (shape:int) maxExp (exps:List<float32[]*float32[]*int*float32*bool*float32>) =
    exps
    |> PSeq.map (fun (st,nst,act,rwd,dn,priority) ->
            {
                State       = st 
                NextState   = nst 
                Action      = act
                Reward      = rwd
                Done        = dn
                Priority    = priority
            })
    |> Seq.truncate maxExp
    |> RandomAccessList.ofSeq


let save path buff =
    failIfEmptyBuffer buff
    let data = 
        match buff with
        | UniformSampled buff -> Map.ofList [0,(exportExperience buff.Buffer)]
        | StratifiedSampled buff -> buff.BufferMap |> Map.map (fun k v -> exportExperience v)
        | PrioritizedSampled buff -> Map.ofList [0,(exportExperience buff.Buffer)]
    let shape,maxExp,minSamples,mode,alpha,priorityEps = 
        match buff with 
        | UniformSampled buff -> (Seq.head buff.Buffer).State.Length, buff.Max, None, "uniform", None, None
        | StratifiedSampled buff -> 
            (buff.BufferMap |> Map.toSeq |> Seq.head |> snd |> Seq.head).State.Length, 
            buff.Max, 
            Some buff.MinSamplesPerStrata,
            "stratified",
            None,
            None
        | PrioritizedSampled buff ->
            (Seq.head buff.Buffer).State.Length,
            buff.Max,
            None,
            "prioritized",
            Some buff.Alpha,
            Some buff.PriorityEps
    let ser = MBrace.FsPickler.BinarySerializer()
    use str = System.IO.File.Create (path:string)
    let sval:VTser = (maxExp,minSamples,shape,mode,alpha,priorityEps,data)
    ser.Serialize(str,sval)

let saveAsync path buff =
    async {
        do save path buff
    }

let load path maxBuff =
    let ser = MBrace.FsPickler.BinarySerializer()
    use str = System.IO.File.OpenRead(path:string)        
    let t1 = DateTime.Now
    printfn $"ExpBuff: loading from {path}"
    let ((maxExp,minSamples,shape,mode,alpha,priorityEps,data):VTser) = ser.Deserialize<VTser>(str)
    let t2 = DateTime.Now
    printfn $"ExpBuff: loaded %0.2f{(t2-t1).TotalMinutes} minutes"
    printfn "ExpBuff: creating tensors"
    let maxExp = maxBuff |> Option.defaultValue maxExp
    let buff =
        match mode, minSamples with 
        | "prioritized", _ -> PrioritizedSampled {Buffer=importExperience shape maxExp data.[0]; Max=maxExp; Alpha=defaultArg alpha 0.6; PriorityEps=defaultArg priorityEps 1e-4}
        | _, None ->  UniformSampled {Buffer=importExperience shape maxExp data.[0]; Max=maxExp}
        | _, Some minSamples -> StratifiedSampled {BufferMap=data |> Map.map (fun k v -> importExperience shape maxExp v); Max=maxExp; MinSamplesPerStrata=minSamples}
    let t3 = DateTime.Now
    printfn $"ExpBuff: random access list created %0.2f{(t3-t2).TotalMinutes} minutes, size {buff.Length()}"
    buff

module Experience
open System
open MathNet.Numerics
open FSharpx.Collections
open FSharp.Collections.ParallelSeq
open TorchSharp
open TorchSharp.Fun
open MathNet.Numerics.Statistics

type Experience = {State:torch.Tensor; NextState:torch.Tensor; Action:int; Reward:float32; Done:bool; Priority:float32}
type ExperienceBufferUniform = {Buffer:RandomAccessList<Experience>; Max:int}
type ExperienceBufferStratified = {BufferMap:Map<int,RandomAccessList<Experience>>; Max:int; MinSamplesPerStrata:int}
type ExperienceBufferPrioritized = {Buffer:RandomAccessList<Experience>; Max:int; Alpha:float; PriorityEps:float}
type ExperienceBuffer =
    | UniformSampled of ExperienceBufferUniform
    | StratifiedSampled of ExperienceBufferStratified
    | PrioritizedSampled of ExperienceBufferPrioritized
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
         
let private appendExperience max newExp (ls:RandomAccessList<Experience>) = 
    if ls.Length < max then 
        RandomAccessList.cons newExp ls
    else
        let idx = torch.randint(ls.Length, [|1|], dtype=torch.int) |> Tensor.getData<int> |> Array.nth 0
        let prev = ls.[idx]
        prev.NextState.Dispose()
        prev.State.Dispose()
        RandomAccessList.update idx newExp ls

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

let private sampleExperience max n (expBuff:RandomAccessList<Experience>) =
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

let private stratifiedSample n (buff:ExperienceBufferStratified) =
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

let private prioritizedSample n beta (buff:ExperienceBufferPrioritized) =
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
            let cdf =
                probabilities
                |> Array.scan (+) 0.0
                |> Array.tail

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

let sample n = function 
    | UniformSampled buff -> sampleExperience buff.Max n buff.Buffer
    | StratifiedSampled buff -> stratifiedSample n buff
    | PrioritizedSampled buff ->
        let sampled, _, _ = prioritizedSample n 0.0 buff
        sampled

let recall (device:torch.Device) n buff =
    failIfEmptyBuffer buff
    let exps = sample n buff
    if Array.isEmpty exps then failwith "cannot recall an empty minibatch from replay buffer"
    //batch sample data into separate arrays/tensors
    let states     = exps |> Array.map _.State.unsqueeze(0L)     |> torch.vstack   
    let states     = states.clone().contiguous()
    let nextStates = exps |> Array.map _.NextState.unsqueeze(0L) |> torch.vstack
    let nextStates = nextStates.clone().contiguous()
    let actions    = exps |> Array.map _.Action
    let rewards    = exps |> Array.map _.Reward
    let dones      = exps |> Array.map _.Done
    let d_states = states.``to`` device
    let d_nextStates = nextStates.``to`` device
    if states.device <> d_states.device then states.Dispose()
    if nextStates.device <> d_nextStates.device then nextStates.Dispose()
    d_states,d_nextStates,rewards,actions,dones

let recallPrioritized (device:torch.Device) n beta = function
    | PrioritizedSampled buff ->
        failIfEmptyBuffer (PrioritizedSampled buff)
        let exps, indices, weights = prioritizedSample n beta buff
        if Array.isEmpty exps then failwith "cannot recall an empty minibatch from prioritized replay buffer"
        let states = exps |> Array.map _.State.unsqueeze(0L) |> torch.vstack
        let states = states.clone().contiguous()
        let nextStates = exps |> Array.map _.NextState.unsqueeze(0L) |> torch.vstack
        let nextStates = nextStates.clone().contiguous()
        let actions = exps |> Array.map _.Action
        let rewards = exps |> Array.map _.Reward
        let dones = exps |> Array.map _.Done
        let d_states = states.``to`` device
        let d_nextStates = nextStates.``to`` device
        if states.device <> d_states.device then states.Dispose()
        if nextStates.device <> d_nextStates.device then nextStates.Dispose()
        d_states,d_nextStates,rewards,actions,dones,indices,weights
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
type Tser = int * int option * int64[] * string * float option * float option * Map<int,List<float32[]*float32[]*int*float32*bool*float32>> 
        //  max * minSamples * tensor shape * replayMode * alpha * eps * buffer payload

let private exportExperience (expBuff:RandomAccessList<Experience>) = 
    expBuff
    |> Seq.choose (fun x-> 
        try
            (x.State.data<float32>().ToArray(),
            x.NextState.data<float32>().ToArray(),
            x.Action,
            x.Reward,
            x.Done,
            x.Priority)
            |> Some
        with ex ->
            printfn "Ser error: %s" ex.Message
            None )
    |> Seq.toList

let private importExperience (shape:int64[]) (exps:List<float32[]*float32[]*int*float32*bool*float32>) =
    exps
    |> PSeq.map (fun (st,nst,act,rwd,dn,priority) ->
            let bst = System.Runtime.InteropServices.MemoryMarshal.Cast<float32,byte>(Span(st))
            let bnst = System.Runtime.InteropServices.MemoryMarshal.Cast<float32,byte>(Span(nst))
            let tst = torch.zeros(shape,dtype=Nullable torch.float32)
            tst.bytes <- bst
            let tnst = torch.zeros(shape,dtype=Nullable torch.float32)
            tnst.bytes <- bnst
            {
                State       = tst
                NextState   = tnst
                Action      = act
                Reward      = rwd
                Done        = dn
                Priority    = priority
            })
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
        | UniformSampled buff -> (Seq.head buff.Buffer).State.shape, buff.Max, None, "uniform", None, None
        | StratifiedSampled buff -> 
            (buff.BufferMap |> Map.toSeq |> Seq.head |> snd |> Seq.head).State.shape, 
            buff.Max, 
            Some buff.MinSamplesPerStrata,
            "stratified",
            None,
            None
        | PrioritizedSampled buff ->
            (Seq.head buff.Buffer).State.shape,
            buff.Max,
            None,
            "prioritized",
            Some buff.Alpha,
            Some buff.PriorityEps
    let ser = MBrace.FsPickler.BinarySerializer()
    use str = System.IO.File.Create (path:string)
    let sval:Tser = (maxExp,minSamples,shape,mode,alpha,priorityEps,data)
    ser.Serialize(str,sval)

let saveAsync path buff =
    async {
        do save path buff
    }

let deserializeBuffer path maxSamples = 
    let ser = MBrace.FsPickler.BinarySerializer()
    use str = System.IO.File.OpenRead(path:string)        
    let t1 = DateTime.Now
    printfn $"ExpBuff: loading from {path}"
    let ((maxExp,minSamples,shape,mode,alpha,priorityEps,data):Tser) = ser.Deserialize<Tser>(str)
    let t2 = DateTime.Now
    let count = data |> Map.toSeq |> Seq.sumBy (fun (x,ys) -> ys.Length)
    printfn $"ExpBuff: loaded {count} in  %0.2f{(t2-t1).TotalMinutes} minutes"
    printfn "ExpBuff: creating tensors"
    let maxExp = maxSamples |> Option.map (fun m -> min m maxExp) |> Option.defaultValue maxExp
    let data = data |> Map.map (fun k v -> v |> List.truncate maxExp)
    (maxExp,minSamples,shape,mode,alpha,priorityEps,data)

let load path maxSamples =
    let (maxExp,minSamples,shape,mode,alpha,priorityEps,data) = deserializeBuffer path maxSamples
    System.GC.Collect()
    let t2 = DateTime.Now
    let buff =
        match mode, minSamples with 
        | "prioritized", _ ->
            PrioritizedSampled {Buffer=importExperience shape data.[0]; Max=maxExp; Alpha=defaultArg alpha 0.6; PriorityEps=defaultArg priorityEps 1e-4}
        | _, None ->  UniformSampled {Buffer=importExperience shape data.[0]; Max=maxExp}
        | _, Some minSamples -> StratifiedSampled {BufferMap=data |> Map.map (fun k v -> importExperience shape v); Max=maxExp; MinSamplesPerStrata=minSamples}
    let t3 = DateTime.Now
    printfn $"ExpBuff: random access list created %0.2f{(t3-t2).TotalMinutes} minutes"
    buff

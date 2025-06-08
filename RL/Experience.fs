module Experience
open System
open MathNet.Numerics
open FSharpx.Collections
open FSharp.Collections.ParallelSeq
open TorchSharp
open TorchSharp.Fun
open MathNet.Numerics.Statistics

type Experience = {State:torch.Tensor; NextState:torch.Tensor; Action:int; Reward:float32; Done:bool}
type ExperienceBufferUniform = {Buffer:RandomAccessList<Experience>; Max:int}
type ExperienceBufferStratified = {BufferMap:Map<int,RandomAccessList<Experience>>; Max:int; MinSamplesPerStrata:int}
type ExperienceBuffer = UniformSampled of ExperienceBufferUniform | StratifiedSampled of ExperienceBufferStratified

let createUniformSampled maxExperiance = UniformSampled {Buffer=RandomAccessList.empty; Max=maxExperiance}

let createStratifiedSampled maxExperincePerStrata minSamplesPerStrata =    
        StratifiedSampled {BufferMap=Map.empty; Max=maxExperincePerStrata; MinSamplesPerStrata=minSamplesPerStrata}
        
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

let private sampleExperience max n (expBuff:RandomAccessList<Experience>) =
    if expBuff.Length <= n then 
        Seq.toArray expBuff 
    else 
        let maxLen = min max  expBuff.Length //temporarily buffer may be longer than max
        let idx = torch.randperm(int64 maxLen,dtype=torch.int) |> Tensor.getData<int> 
        [|for i in 0..n-1 -> expBuff.[idx.[i]]|]

let sample n = function 
    | UniformSampled buff -> sampleExperience buff.Max n buff.Buffer
    | StratifiedSampled buff -> 
        let minSamples = buff.MinSamplesPerStrata
        let keys = buff.BufferMap.Keys |> Seq.sort |> Seq.toArray
        let strataCounts = keys |> Array.map (fun k -> float buff.BufferMap.[k].Length)
        let samplesPerStrata = //sample count for each strata proportional to each strata's size; where the sum ~= n; ensures some representation of each action type in batch
            LinearAlgebra.Double.Vector.Build.DenseOfEnumerable(strataCounts).Normalize(1.0).ToArray() 
            |> Array.map (fun x -> int(float n * x))
        let keyedSamples = Array.zip keys samplesPerStrata |> Array.sortBy snd  //lowest samples first
        let expSamples1,rem =
            (([||],n),keyedSamples |> Array.take (keyedSamples.Length - 1)) 
            ||> Array.fold (fun (acc,rem) (k,p) -> 
                let p = max minSamples p                
                let exps = sampleExperience buff.Max p (buff.BufferMap.[k])
                Array.append acc exps,rem - p)
        let expSamples2 = sampleExperience buff.Max rem buff.BufferMap.[fst (Array.last keyedSamples)] //take remaining samples (out of n) from the largest strata
        let expSamples = Array.append expSamples1 expSamples2 |> Array.randomShuffle
        expSamples

let recall (device:torch.Device) n buff =
    let exps = sample n buff
    //batch sample data into separate arrays/tensors
    let states     = exps |> Array.map _.State.unsqueeze(0L)     |> torch.vstack        
    let nextStates = exps |> Array.map _.NextState.unsqueeze(0L) |> torch.vstack
    let actions    = exps |> Array.map _.Action
    let rewards    = exps |> Array.map _.Reward
    let dones      = exps |> Array.map _.Done
    let d_states = states.``to`` device
    let d_nextStates = states.``to`` device
    if states.device <> d_states.device then states.Dispose()
    if nextStates.device <> d_nextStates.device then nextStates.Dispose()
    d_states,d_nextStates,rewards,actions,dones

//use built-in F# types for serialization - works better for .fsx scripts
type Tser = int * int option * int64[]      * Map<int,List<float32[]*float32[]*int*float32*bool>> 
        //  max * minSamples * tensor shape * buffer action->experience (if map has only one value it is UniformSampled)

let private exportExperience (expBuff:RandomAccessList<Experience>) = 
    expBuff
    |> Seq.map (fun x-> 
            x.State.data<float32>().ToArray(),
            x.NextState.data<float32>().ToArray(),
            x.Action,
            x.Reward,
            x.Done)
    |> Seq.toList

let private importExperience (shape:int64[]) (exps:List<float32[]*float32[]*int*float32*bool>) =
    exps
    |> PSeq.map (fun (st,nst,act,rwd,dn) ->
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
            })
    |> RandomAccessList.ofSeq


let save path buff =
    let data = 
        match buff with
        | UniformSampled buff -> Map.ofList [0,(exportExperience buff.Buffer)]
        | StratifiedSampled buff -> buff.BufferMap |> Map.map (fun k v -> exportExperience v)
    if Map.isEmpty data then failwithf "empty buffer cannot be saved as tensor shape is unknown"
    let shape,maxExp,minSamples = 
        match buff with 
        | UniformSampled buff -> (Seq.head buff.Buffer).State.shape, buff.Max, None
        | StratifiedSampled buff -> 
            (buff.BufferMap |> Map.toSeq |> Seq.head |> snd |> Seq.head).State.shape, 
            buff.Max, 
            Some buff.MinSamplesPerStrata
    let ser = MBrace.FsPickler.BinarySerializer()
    use str = System.IO.File.Create (path:string)
    let sval:Tser = (maxExp,minSamples,shape,data)
    ser.Serialize(str,sval)

let saveAsync path buff =
    async {
        do save path buff
    }

let load path =
    let ser = MBrace.FsPickler.BinarySerializer()
    use str = System.IO.File.OpenRead(path:string)        
    let t1 = DateTime.Now
    printfn $"ExpBuff: loading from {path}"
    let ((maxExp,minSamples,shape,data):Tser) = ser.Deserialize<Tser>(str)
    let t2 = DateTime.Now
    printfn $"ExpBuff: loaded %0.2f{(t2-t1).TotalMinutes} minutes"
    printfn "ExpBuff: creating tensors"
    let buff =
        match minSamples with 
        | None ->  UniformSampled {Buffer=importExperience shape data.[0]; Max=maxExp}
        | Some minSamples -> StratifiedSampled {BufferMap=data |> Map.map (fun k v -> importExperience shape v); Max=maxExp; MinSamplesPerStrata=minSamples}
    let t3 = DateTime.Now
    printfn $"ExpBuff: random access list created %0.2f{(t3-t2).TotalMinutes} minutes"
    buff

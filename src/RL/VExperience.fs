module VExperience

open System
open MathNet.Numerics
open FSharpx.Collections
open FSharp.Collections.ParallelSeq
open TorchSharp
open TorchSharp.Fun

type VExperience = {State:float[]; NextState:float[]; Action:int; Reward:float; Done:bool}
type VExperienceBufferUniform = {Buffer:RandomAccessList<VExperience>; Max:int}
type VExperienceBufferStratified = {BufferMap:Map<int,RandomAccessList<VExperience>>; Max:int; MinSamplesPerStrata:int}
type VExperienceBuffer = UniformSampled of VExperienceBufferUniform | StratifiedSampled of VExperienceBufferStratified

let createUniformSampled maxExperiance = UniformSampled {Buffer=RandomAccessList.empty; Max=maxExperiance}

let createStratifiedSampled maxExperincePerStrata minSamplesPerStrata =    
        StratifiedSampled {BufferMap=Map.empty; Max=maxExperincePerStrata; MinSamplesPerStrata=minSamplesPerStrata}
        
let private appendExperience max exp (ls:RandomAccessList<VExperience>) = 
    let ls = RandomAccessList.cons exp ls
    if ls.Length > max * 2 then
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

let private sampleExperience max n (expBuff:RandomAccessList<VExperience>) =
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
        let counts = keys |> Array.map (fun k -> float buff.BufferMap.[k].Length)
        let samples = 
            LinearAlgebra.Double.Vector.Build.DenseOfEnumerable(counts).Normalize(1.0).ToArray() 
            |> Array.map (fun x -> int(float n * x))
        let keyedSamples = Array.zip keys samples |> Array.sortBy snd  //lowest samples first
        let expSamples1,rem =
            (([||],n),keyedSamples |> Array.take (keyedSamples.Length - 1)) 
            ||> Array.fold (fun (acc,rem) (k,p) -> 
                let p = max minSamples p                
                let exps = sampleExperience buff.Max p (buff.BufferMap.[k])
                Array.append acc exps,rem - p)
        let expSamples2 = sampleExperience buff.Max rem buff.BufferMap.[fst (Array.last keyedSamples)]
        let expSamples = Array.append expSamples1 expSamples2 |> Array.randomShuffle
        expSamples

let recall n buff =
    let exps = sample n buff
    let states     = exps |> Array.map (fun x->x.State)
    let nextStates = exps |> Array.map (fun x->x.NextState)
    let actions    = exps |> Array.map (fun x->x.Action)
    let rewards    = exps |> Array.map (fun x -> x.Reward)
    let dones      = exps |> Array.map (fun x->x.Done)
    states,nextStates,rewards,actions,dones

//use built-in F# types for serialization - works better for .fsx scripts
type VTser = int * int option * int      * Map<int,List<float[]*float[]*int*float*bool>> 
        //  max * minSamples * tensor shape * buffer action->experience (if map has only one value it is UniformSampled)

let private exportExperience (expBuff:RandomAccessList<VExperience>)  = 
    expBuff
    |> Seq.map (fun x-> 
            x.State,
            x.NextState,
            x.Action,
            x.Reward,
            x.Done)
    |> Seq.toList

let private importExperience (shape:int) (exps:List<float[]*float[]*int*float*bool>) =
    exps
    |> PSeq.map (fun (st,nst,act,rwd,dn) ->
            {
                State       = st
                NextState   = nst
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
        | UniformSampled buff -> (Seq.head buff.Buffer).State.Length, buff.Max, None
        | StratifiedSampled buff -> 
            (buff.BufferMap |> Map.toSeq |> Seq.head |> snd |> Seq.head).State.Length, 
            buff.Max, 
            Some buff.MinSamplesPerStrata
    let ser = MBrace.FsPickler.BinarySerializer()
    use str = System.IO.File.Create (path:string)
    let sval:VTser = (maxExp,minSamples,shape,data)
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
    let ((maxExp,minSamples,shape,data):VTser) = ser.Deserialize<VTser>(str)
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

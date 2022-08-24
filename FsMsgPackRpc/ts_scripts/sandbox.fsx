#load "../scripts/packages.fsx"
#load "../TsData.fs"
#load "../RL.fs"
open System.Threading.Tasks
open TorchSharp
open TorchSharp.Fun
open TsData
open RL

let fn = @"E:\s\tradestation\mes_5_min.bin"

let data = TsData.loadBars fn

let dataChunks = 
        data 
        |> Seq.map (fun b -> [|b.Open; b.High; b.Low; b.Close; b.Volume|])
        |> Seq.chunkBySize 40 


let prev =  dataChunks |> Seq.skip 2 |> Seq.head 
let n = dataChunks |> Seq.skip 3 |> Seq.head |> Array.item 0

let i1 = torch.TensorIndex.Single(1)
let ``...`` = torch.TensorIndex.Ellipsis
let ``:`` = torch.TensorIndex.Colon
let skipHead = torch.TensorIndex.Slice(1L)

let ts = torch.tensor(prev |> Seq.collect (fun x->x) |> Seq.toArray,dtype=torch.float32).reshape(40,-1)
let tn = torch.tensor(n, dtype=torch.float32)
let ts2 = torch.vstack([|ts;tn|])

let ts3 = if ts.shape.[0] > 40L then ts.index(skipHead) else ts  // 40 x 5 


let t1inp = 
    let r1 = 
        dataChunks
        |> Seq.take 10
        |> Seq.collect(Seq.collect (fun x->x))
        |> Seq.toArray
    torch.tensor(r1,dtype=torch.float32).reshape(-1,40,5)


let c1 = torch.nn.Conv1d(40L,5L,4L,stride=2L)
let t1c1 = c1.forward(t1inp).flatten(1L)
let l2 = torch.nn.Linear(5L,2L)
let tout = l2.forward(t1c1)

let mt =
        torch.nn.Conv1d(40L, 64L, 4L, stride=2L, padding=3L)     //b x 64L x 4L   
        ->> torch.nn.BatchNorm1d(64L)
        ->> torch.nn.Dropout(0.3)
        ->> torch.nn.ReLU()
        ->> torch.nn.Conv1d(64L,64L,3L)
        ->> torch.nn.ReLU()
        ->> torch.nn.Flatten()
        ->> torch.nn.Linear(128L,2L)

let actionIdx (actions:torch.Tensor) = 
    [|
        torch.TensorIndex.Tensor (torch.arange(actions.shape.[0], dtype=torch.int64))            //batch dimension
        torch.TensorIndex.Tensor (actions) //actions dimension
    |]


let acts = mt.forward(t1inp)
let macts = acts.argmax(1L)
let t_acts = Tensor.getData<float32>(acts)
let t_macts = Tensor.getData<int64>(macts)
let qn = actionIdx macts
let q = acts.index (qn)
let d_q = Tensor.getData<float32>(q)

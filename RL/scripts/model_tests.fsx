#load "packages.fsx"
open AirSimCar
open TorchSharp
open TorchSharp.Fun
open System.IO
open DQN

let root = System.Environment.GetEnvironmentVariable("AIRSIM_DQN")
let (@@) a b = Path.Combine(a,b)

//DQN pytorch model
let createModel () = 
    torch.nn.Conv2d(1L,32L,8L,stride=4L)
    ->> torch.nn.ReLU()
    ->> torch.nn.Conv2d(32L,64L,4L,stride=2L)
    ->> torch.nn.ReLU()
    ->> torch.nn.Conv2d(64L,64L,3L,stride=1L)
    ->> torch.nn.ReLU()
    ->> torch.nn.Flatten()
    ->> torch.nn.Linear(3136L,512L)
    ->> torch.nn.ReLU()
    ->> torch.nn.Linear(512L,CarEnvironment.discreteActions)

let modelFile = root @@ "DQN_airsim_test1.bin"

let mm  = DQNModel.create createModel
DQNModel.save  modelFile mm
let mm2 = DQNModel.load createModel modelFile

mm.Online.Module.save(modelFile)

let d_mm = mm.Online.Module.state_dict()
let d_mm2 = mm2.Online.Module.state_dict()

Seq.zip d_mm d_mm2 |> Seq.forall (fun (a,b) -> a.Key=b.Key && a.Value.shape = b.Value.shape)


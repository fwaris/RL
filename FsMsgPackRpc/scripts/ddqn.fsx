#load "packages.fsx"
open System
open AirSimCar
open CarEnvironment
open TorchSharp
open TorchSharp.Fun
open DDQN

//ddqn pytorch model
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

let model = DDQNModel.create createModel
let device = torch.CPU
let gamma = 0.9f
let exploration = {Rate=1.0; Decay=0.99999975; Min=0.1}
let initDDQN = DDQN.create model gamma exploration CarEnvironment.discreteActions device
let initExperience = Experience.createBuffer 100000
let batchSize = 32
let lossFn = torch.nn.functional.smooth_l1_loss()

let opt = torch.optim.Adam(model.Online.Module.parameters(), lr=0.00025)

let updateQ td_estimate td_target =
    use loss = lossFn.Invoke(td_estimate,td_target)
    opt.zero_grad()
    loss.backward()
    use t = opt.step() 
    loss.item()

let trainDDQN (go:bool ref) =
    let c = new CarClient(AirSimCar.Defaults.options)
    c.Connect(AirSimCar.Defaults.address,AirSimCar.Defaults.port)
    initCar c |> Async.AwaitTask |> Async.RunSynchronously

    let initState = RLState.Default
    let initCtrls = {CarControls.Default with throttle = 1.0}
    let burnIn = 100
    let learnEvery = 3
    let syncEvery = 100
    
    let rec loop count state ctrls ddqn experienceBuff =
        async {
            try
                let action,ddqn = DDQN.act state.DepthImage ddqn
                let! (state,ctrls,reward,isDone) = step c (state,ctrls) action |> Async.AwaitTask
                printfn $"reward: {reward}, isDone: {isDone}"
                let experience = {NextState = state.DepthImage; Action=action; State = state.PrevDepthImage; Reward=float32 reward; Done=isDone <> NotDone}
                let experienceBuff = Experience.append experience experienceBuff  
                if not go.Value then
                    //do! initCar c |> Async.AwaitTask
                    c.Disconnect()
                    printfn "stopped"
                else
                    let count = count + 1
                    match isDone with NotDone -> () | _ ->  do! c.reset() |> Async.AwaitTask
                    


                    match isDone with
                        | NotDone ->                 
                            return! loop (count+1) state ctrls ddqn experienceBuff 
                        | doneRsn ->
                            printfn $"Done: {doneRsn}"                        
                           
                            do! Async.Sleep 1000 // need to wait for the car to settle down after reset
                            return! loop initState initCtrls experienceBuff initAction
            with ex -> printfn "%A" ex.Message
        }
    loop initState initCtrls initDDQN initExperience

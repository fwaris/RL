open AirSimCar
open TorchSharp
open TorchSharp.Fun
open System
open System.IO
open DQN
open System.Threading.Tasks
open VExperience
open System.Numerics
open FSharp.Control

let carIds = ["Car1"; "Car2"] //"PhysXCar"
//let carIds = ["Car1"] //"PhysXCar"
//let carIds = ["PhysXCar"]

let burnInMax = 1000
let learnEvery = 4
let syncEvery = 300
let saveBuffEvery = 5000
let episodeLength = 30
let BUFF_MAX = 30_000

let (@@) a b = Path.Combine(a,b)
let root = System.Environment.GetEnvironmentVariable("DATA_DRIVE") @@ "s/ddqn"

//DQN pytorch model
let createModel3d () = 
    torch.nn.Conv3d(1L,8L,struct (1L,8L,8L),stride=struct(1L,4L,4L))
    ->> torch.nn.InstanceNorm3d(8L)
    ->> torch.nn.ReLU()
    ->> torch.nn.Conv3d(8L,16L,struct (1L,4L,4L),stride=struct(1L,2L,2L))
    ->> torch.nn.ReLU()
    ->> torch.nn.Flatten()
    ->> torch.nn.Linear(57600,512L)
    ->> torch.nn.LayerNorm(512L)
    ->> torch.nn.ReLU()
    ->> torch.nn.Linear(512L,CarEnvironment.discreteActions)

let modelFile = root @@ "DQN_airsim.bin"
let exprFile = root @@ "expr_buff_airsim.bin"
let device = torch.CUDA
let model = 
    if File.Exists modelFile then         //restart session
        DQNModel.load createModel3d modelFile
    else
        DQNModel.create createModel3d
let m11 = model.Online.Module.``to`` device //|> ignore
let m21 = model.Target.Module.``to`` device //|> ignore
let m1 = model.Online.Module
let n2 = model.Online.Module

let initExperience = lazy(
    if File.Exists exprFile then          //reuse saved buffer
        printfn $"loading experience buffer from file {exprFile}"
        VExperience.load exprFile (Some BUFF_MAX)
    else
        VExperience.createUniformSampled BUFF_MAX)

let burnIn = lazy(burnInMax - initExperience.Value.Length() |> max 0)
let lossFn = torch.nn.SmoothL1Loss()
let gamma = 0.9f
let minExpRate = 0.01
let initExpRate = 0.2
let exploration = {Exploration.Default with Min=minExpRate}
let initDQN = DQN.create model gamma exploration CarEnvironment.discreteActions
let batchSize = 32
let opt = torch.optim.RAdam(model.Online.Module.parameters(), lr=0.0005)

let updateQ td_estimate td_target =
    use loss = lossFn.forward(td_estimate,td_target)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.Online.Module.parameters(),299.0) |> ignore
    use t = opt.step() 
    loss.ToDouble()

let learn step (state:CarEnvironment.RLState) (dqn:DQN) experienceBuff = 
    async {
        printfn "l: start"
        let device = DQNModel.device model
        let states,nextStates,rewards,actions,dones = VExperience.recall batchSize experienceBuff  //sample from experience((
        let sRaw = Array.collect id states
        let nRaw = Array.collect id nextStates
        let expLen = states.Length * 3 * 256 * 256 
        if nRaw.Length = sRaw.Length && sRaw.Length <> expLen then 
            let i  = 1
            ()
        let batchShape = Seq.append [int64 states.Length] state.CarStates.[0].CombinedImage.shape |> Seq.toArray
        use states = torch.tensor(sRaw, device=device).reshape(batchShape)
        use nextStates = torch.tensor(nRaw, device=device).reshape(batchShape)

        try
            let td_est = DQN.td_estimate states actions dqn.Model.Online
            //let td_est_d = td_est.data<float32>().ToArray() //DQN invocations 
            let td_tgt = DQN.td_target rewards nextStates dones dqn
            let loss = updateQ td_est td_tgt  //update online model 
            printfn $"{step.Num}, loss: {loss}"
            System.GC.Collect()                            
            printfn "l: end"
        with ex -> 
            raise ex
    } 

let trainDQN (clnt:CarClient) (logLevel:CarEnvironment.LogLevel ref) (go:bool ref) =    
    CarEnvironment.enableApiControl carIds clnt |> Async.RunSynchronously
    let initState = CarEnvironment.RLState.Create carIds
    let rng = System.Random()
    let rec loop (step:Step) (state:CarEnvironment.RLState) (dqn:DQN) (experienceBuff:VExperienceBuffer) =
        async {
            try
                //select action to take
                let actions = 
                    if step.Num <= burnIn.Value then
                        state.CarStates |> List.map (fun _ -> rng.Next(CarEnvironment.discreteActions))        //select random actions in the beginning to build the experience buffer
                    else
                       state.CarStates |> List.map(fun s -> fst (DQN.selectAction s.CombinedImage dqn step))   //select policy driven actions

                //perform actions in environment, observe new states, compute rewards
                let! state = CarEnvironment.step logLevel clnt state actions
                if logLevel.Value.isVerbose() then
                    printfn $"{step.Num}, exp:exp: %.03f{step.ExplorationRate}, buff={experienceBuff.Length()}"
                state.CarStates |> List.iter(fun s -> 
                    let action = s.ActionHistory |> List.tryHead |> Option.defaultValue 0
                    let ctrls = s.Controls
                    if logLevel.Value.isVerbose() then
                        printfn $"{step.Num}, exp:exp: %.03f{step.ExplorationRate}, buff={experienceBuff.Length()}"
                        printfn $"{s.CarId} reward: %0.2f{s.Reward}, isDone: {state.EpisodeSteps}-{s.DoneReason}, {action}, s,b,t=({ctrls.steering},{ctrls.brake},{ctrls.throttle}), spd=%0.02f{s.Speed}")

                let experienceBuff = 
                    (experienceBuff,List.zip state.CarStates actions)
                    ||> List.fold(fun buff (s,a) ->
                        //add to experience buffer
                        let experience : VExperience =                     
                                {
                                    NextState = s.CombinedImage |> Tensor.getData<float32>;
                                    Action=a; 
                                    State = s.PrevCombinedImage |> Tensor.getData<float32>; 
                                    Reward=float32 s.Reward; 
                                    Done=s.DoneReason <> CarEnvironment.NotDone
                                    Priority=1.0f
                                }
                        VExperience.append experience buff)

                //check for termination
                if not go.Value then
                    printfn "stopped"
                else
                    if step.Num > burnIn.Value && step.Num % learnEvery = 0 then            
                        do! learn step state dqn experienceBuff
                    if step.Num > 0 && step.Num % saveBuffEvery = 0 then
                        do! clnt.simPause(true) |> Async.AwaitTask
                        VExperience.saveAsync exprFile experienceBuff |> Async.Start
                        do! clnt.simPause(false) |> Async.AwaitTask

                    //periodically sync target model with online model
                    if step.Num > 0 && step.Num % syncEvery = 0 then 
                        do! clnt.simPause(true) |> Async.AwaitTask
                        DQNModel.save modelFile dqn.Model
                        DQNModel.sync dqn.Model 
                        do! clnt.simPause(false) |> Async.AwaitTask
                        printfn $"Exploration rate: {step.ExplorationRate}"

                    let! state = 
                        if state.EpisodeSteps = 0 then 
                            async{
                                let! carStates =  CarEnvironment.resetCars clnt state.CarStates
                                do! Async.Sleep 500 // give the cars time to settle.
                                return {state with CarStates = carStates}
                            }
                        else 
                            async{return state}
                    
                    return! loop (DQN.updateStep dqn.Exploration step) state dqn experienceBuff
                    
            with ex -> printfn "trainDQN: %A" (ex.Message,ex.StackTrace)
        }
    loop ({Num=0; ExplorationRate=initExpRate}) initState initDQN initExperience.Value

let runDQN (clnt:CarClient) (logLevel:CarEnvironment.LogLevel ref) (go:bool ref) =    
    CarEnvironment.enableApiControl carIds clnt |> Async.RunSynchronously
    let initState = CarEnvironment.RLState.Create carIds
    let rec loop (state:CarEnvironment.RLState) (dqn:DQN)  =
        async {
            try
                //select action to take
                let actions = state.CarStates |> List.map(fun s -> fst (DQN.bestAction s.CombinedImage dqn))   //select policy driven actions

                //perform actions in environment, observe new states, compute rewards
                let! state = CarEnvironment.stepRun logLevel clnt state actions
                state.CarStates |> List.iter(fun s -> 
                    let action = s.ActionHistory |> List.tryHead |> Option.defaultValue 0
                    let ctrls = s.Controls
                    printfn $"{s.CarId} reward: %0.2f{s.Reward}, isDone: {state.EpisodeSteps}-{s.DoneReason}, {action}, s,b,t=({ctrls.steering},{ctrls.brake},{ctrls.throttle}), spd=%0.02f{s.Speed}")

                //check for termination
                if not go.Value then
                    printfn "stopped"
                else                    
                    return! loop state dqn
                    
            with ex -> printfn "trainDQN: %A" (ex.Message,ex.StackTrace)
        }
    loop initState initDQN

let runTraining doLog go  =
    async {
        let c = new CarClient(AirSimCar.Defaults.options)
        c.Connect(AirSimCar.Defaults.address,AirSimCar.Defaults.port)      
        do! trainDQN c doLog go

    }

let runSim doLog go  =
    async {
        let c = new CarClient(AirSimCar.Defaults.options)
        c.Connect(AirSimCar.Defaults.address,AirSimCar.Defaults.port)      
        do! runDQN c doLog go

    }

let go = ref true
let logLevel = ref CarEnvironment.Verbose

System.Console.WriteLine("r = runSim, any other key to train model")
let k = System.Console.ReadKey()
match k.KeyChar with
| 'r' -> runSim logLevel go |> Async.Start
|  _  -> runTraining logLevel go  |> Async.Start

let printUsage() = System.Console.WriteLine("x=quit,s=save model;q,v=loglevel;c=approach corner toggle")
printUsage()
let rec loop carId = 
    let k = System.Console.ReadKey()
    match k.KeyChar with 
    | 'x' -> go.Value <- false
    | 's' -> DQNModel.save modelFile model
             loop carId
    | 'v' -> logLevel.Value <-CarEnvironment.Verbose
             loop carId
    | 'q' -> logLevel.Value <- CarEnvironment.Quite
             loop carId
    | _ -> printUsage(); loop carId
loop carIds.[0]


(*
System.Runtime.GCSettings.IsServerGC

)
logLevel.Value <- CarEnvironment.Verbose
logLevel.Value <- CarEnvironment.Quite

DQNModel.save modelFile model
go.Value <- false
System.Runtime.GCSettings.IsServerGC
84*84*4
*)


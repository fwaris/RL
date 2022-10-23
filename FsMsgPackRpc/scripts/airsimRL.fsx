#load "packages.fsx"
open AirSimCar
open TorchSharp
open TorchSharp.Fun
open System.IO
open DQN
open System.Threading.Tasks

let burnInMax = 200000
let learnEvery = 4
let syncEvery = 10000
let saveBuffEvery = 50000

let (@@) a b = Path.Combine(a,b)
let root = System.Environment.GetEnvironmentVariable("DATA_DRIVE") @@ "s/ddqn"

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

let modelFile = root @@ "DQN_airsim.bin"
let exprFile = root @@ "expr_buff_airsim.bin"
let model = 
    if File.Exists modelFile then         //restart session
        DQNModel.load createModel modelFile
    else
        DQNModel.create createModel
let BUFF_MAX = 500_000
let initExperience =
    if File.Exists exprFile then          //reuse saved buffer
        printfn $"loading experience buffer from file {exprFile}"
        {Experience.load exprFile with Max = BUFF_MAX}
    else
        Experience.createBuffer BUFF_MAX
let burnIn = burnInMax - initExperience.Buffer.Length |> max 0
let lossFn = torch.nn.SmoothL1Loss()
let device = torch.CUDA
let gamma = 0.9f
let minExpRate = 0.01
let initExpRate = 0.1 
let exploration = {Decay=0.9999; Min=minExpRate}
let initDQN = DQN.create model gamma exploration CarEnvironment.discreteActions device
let batchSize = 32
let opt = torch.optim.Adam(model.Online.Module.parameters(), lr=0.00025)

let updateQ td_estimate td_target =
    use loss = lossFn.forward(td_estimate,td_target)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.Online.Module.parameters(),10.0) |> ignore
    use t = opt.step() 
    loss.ToDouble()

let initCar (clnt:CarClient) = 
    task {
        let! _ = clnt.enableApiControl(true) 
        let! isApi = clnt.isApiControlEnabled() 
        if isApi then
            let! _ = clnt.armDisarm(true) 
            do!  clnt.reset()
        else
            return failwith "unable to put car in api mode"
        do! Async.Sleep 10
    }

let resetCar (clnt:CarClient) = 
    task {
        try
            do! clnt.reset()
            do! Async.Sleep 10
            do! clnt.setCarControls({CarControls.Default with throttle = 0.1})
            let! _ = clnt.simSetObjectPose(CarEnvironment.carId,CarEnvironment.randPose(),true) 
            ()
        with ex ->
            printfn $"resetCar error: {ex.Message}, {ex.StackTrace}"
    }



let trainDQN (clnt:CarClient) (logLevel:CarEnvironment.LogLevel ref) (go:bool ref) =
    initCar clnt |> Async.AwaitTask |> Async.RunSynchronously
    let initState = CarEnvironment.RLState.Default
    let initCtrls = {CarControls.Default with throttle = 1.0}
    let rng = System.Random()
    let rec loop (step:Step) (state:CarEnvironment.RLState) ctrls (dqn:DQN) experienceBuff =
        async {
            try
                //select action to take
                let action = 
                    if step.Num <= burnIn then
                        rng.Next(CarEnvironment.discreteActions)           //select random actions in the beginning to build the experience buffer
                    else
                        DQN.selectAction state.DepthImage dqn step                //select policy action

                //perform action in environment, observe new state, compute reward
                let! (state,ctrls,reward,isDone) = CarEnvironment.step logLevel clnt (state,ctrls) action 500 |> Async.AwaitTask
                printfn $"{step.Num},exp: %.03f{step.ExplorationRate}, reward: {reward}, isDone: {isDone}, {action}, s,b,t=({ctrls.steering},{ctrls.brake},{ctrls.throttle}), spd=%0.02f{state.Speed}, buff={experienceBuff.Buffer.Length}"

                //add to experience buffer
                let experience = {NextState = state.DepthImage; Action=action; State = state.PrevDepthImage; Reward=float32 reward; Done=isDone <> CarEnvironment.NotDone}
                let experienceBuff = Experience.append experience experienceBuff  

                //check for termination
                if not go.Value then
                    printfn "stopped"
                else
                    let updateModelTask =
                    //periodically train online model from a sample of the experience buffer
                        if step.Num > burnIn && step.Num % learnEvery = 0 then                      
                            task {
                                let states,nextStates,rewards,actions,dones = Experience.recall batchSize experienceBuff  //sample from experience
                                use states = states.``to``(dqn.Device)
                                use nextStates = nextStates.``to``(dqn.Device)
                                let td_est = DQN.td_estimate states actions dqn
                                //let td_est_d = td_est.data<float32>().ToArray() //DQN invocations
                                let td_tgt = DQN.td_target rewards nextStates dones dqn
                                let loss = updateQ td_est td_tgt                                                          //update online model 
                                printfn $"{step.Num}, loss: {loss}"
                                System.GC.Collect()                            
                            } 
                        else
                            task{return ()}                        

                    if step.Num > 0 && step.Num % saveBuffEvery = 0 then
                        updateModelTask.Wait()
                        Experience.saveAsync exprFile experienceBuff |> Async.Start

                    //periodically sync target model with online model
                    if step.Num > 0 && step.Num % syncEvery = 0 then 
                        updateModelTask.Wait()
                        DQNModel.save modelFile dqn.Model                        
                        DQNModel.sync dqn.Model dqn.Device
                        printfn $"Exploration rate: {step.ExplorationRate}"
                    let state = 
                        match isDone with 
                        | CarEnvironment.NotDone -> state
                        | _ ->  resetCar clnt |> Async.AwaitTask |> Async.RunSynchronously
                                {state with WasReset=true}
                    
                    return! loop (DQN.updateStep dqn.Exploration step) state ctrls dqn experienceBuff
                    
            with ex -> printfn "trainDQN: %A" (ex.Message,ex.StackTrace)
        }
    loop ({Num=0; ExplorationRate=initExpRate}) initState initCtrls initDQN initExperience


let runTraining doLog go =
    async {
        let c = new CarClient(AirSimCar.Defaults.options)
        c.Connect(AirSimCar.Defaults.address,AirSimCar.Defaults.port)      
        do! trainDQN c doLog go 
    }

(*
System.Runtime.GCSettings.IsServerGC
let go = ref true
let logLevel = ref CarEnvironment.Quite
runTraining logLevel go |> Async.Start

logLevel.Value <- CarEnvironment.Verbose
logLevel.Value <- CarEnvironment.Quite

DQNModel.save modelFile model
go.Value <- false
System.Runtime.GCSettings.IsServerGC

*)


#load "packages.fsx"
open AirSimCar
open TorchSharp
open TorchSharp.Fun
open System.IO
open DDQN

let root = System.Environment.GetEnvironmentVariable("AIRSIM_DDQN")
let (@@) a b = Path.Combine(a,b)

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

let modelFile = root @@ "ddqn_airsim.bin"
let exprFile = root @@ "expr_buff_airsim.bin"
let model = 
    if File.Exists modelFile then         //restart session
        DDQNModel.load createModel modelFile
    else
        DDQNModel.create createModel
let BUFF_MAX = 500_000
let initExperience =
    if File.Exists exprFile then          //reuse saved buffer
        printfn $"loading experience buffer from file {exprFile}"
        {Experience.load exprFile with Max = BUFF_MAX}
    else
        Experience.createBuffer BUFF_MAX
let lossFn = torch.nn.functional.smooth_l1_loss()
let device = torch.CUDA
let gamma = 0.9f
let exploration = {Rate=0.5; Decay=0.9999; Min=0.1}
let initDDQN = DDQN.create model gamma exploration CarEnvironment.discreteActions device
let batchSize = 32
let opt = torch.optim.Adam(model.Online.Module.parameters(), lr=0.00025)

let updateQ td_estimate td_target =
    use loss = lossFn.Invoke(td_estimate,td_target)
    opt.zero_grad()
    loss.backward()
    use t = opt.step() 
    loss.ToDouble()

let resetCar (clnt:CarClient) = 
    task {
        let! _ = clnt.enableApiControl(true) 
        let! isApi = clnt.isApiControlEnabled() 
        if isApi then
            let! _ = clnt.armDisarm(true) 
            let! _ = clnt.simSetObjectPose(CarEnvironment.carId,CarEnvironment.randPose(),true)
            ()
            //do! clnt.setCarControls({CarControls.Default with throttle = 1.0})
        else
            return failwith "unable to put car in api mode"
        do! Async.Sleep 10
    }

let burnInMax = 100000
let burnIn = burnInMax - initExperience.Buffer.Length |> max 0
let learnEvery = 3
let syncEvery = 10000
let saveBuffEvery = 5000

let trainDDQN (clnt:CarClient) (go:bool ref) =
    resetCar clnt |> Async.AwaitTask |> Async.RunSynchronously
    let initState = CarEnvironment.RLState.Default
    let initCtrls = {CarControls.Default with throttle = 1.0}
    let rng = System.Random()
    let rec loop count (state:CarEnvironment.RLState) ctrls ddqn experienceBuff =
        async {
            try
                //select action to take
                let action,ddqn = 
                    if count <= burnIn then
                        rng.Next(CarEnvironment.discreteActions),ddqn           //select random actions in the beginning to build the experience buffer
                    else
                        DDQN.selectAction state.DepthImage ddqn                 //select policy action

                //perform action in environment, observe new state, compute reward
                let! (state,ctrls,reward,isDone) = CarEnvironment.step clnt (state,ctrls) action 1000 |> Async.AwaitTask
                printfn $"{count},exp: %.03f{ddqn.Step.ExplorationRate}, reward: {reward}, isDone: {isDone}, {action}, s,b,t=({ctrls.steering},{ctrls.brake},{ctrls.throttle}), spd=%0.02f{state.Speed}, buff={experienceBuff.Buffer.Length}"

                //add to experience buffer
                let experience = {NextState = state.DepthImage; Action=action; State = state.PrevDepthImage; Reward=float32 reward; Done=isDone <> CarEnvironment.NotDone}
                let experienceBuff = Experience.append experience experienceBuff  

                //check for termination
                if not go.Value then
                    printfn "stopped"
                else
                    //periodically train online model from a sample of the experience buffer
                    if count > burnIn && count % learnEvery = 0 then                      
                        let states,nextStates,rewards,actions,dones = Experience.recall batchSize experienceBuff  //sample from experience
                        use states = states.``to``(ddqn.Device)
                        use nextStates = nextStates.``to``(ddqn.Device)
                        let td_est = DDQN.td_estimate states actions ddqn        
                        //let td_est_d = td_est.data<float32>().ToArray() //ddqn invocations
                        let td_tgt = DDQN.td_target rewards nextStates dones ddqn
                        let loss = updateQ td_est td_tgt                                                          //update online model 
                        printfn $"{count}, loss: {loss}"

                    if count > 0 && count % saveBuffEvery = 0 then
                        Experience.save exprFile experienceBuff 

                    //periodically sync target model with online model
                    if count > 0 && count % syncEvery = 0 then 
                        DDQNModel.save modelFile ddqn.Device ddqn.Model                        
                        DDQNModel.sync ddqn.Model ddqn.Device
                        printfn $"Exploration rate: {ddqn.Step.ExplorationRate}"

                    let count = count + 1
                    match isDone with CarEnvironment.NotDone -> () | _ ->  do! resetCar clnt |> Async.AwaitTask
                    
                    return! loop count state ctrls ddqn experienceBuff
                    
            with ex -> printfn "trainDDQN: %A" (ex.Message,ex.StackTrace)
        }
    loop 0 initState initCtrls initDDQN initExperience


let runTraining go =
    async {
        let c = new CarClient(AirSimCar.Defaults.options)
        c.Connect(AirSimCar.Defaults.address,AirSimCar.Defaults.port)      
        do! trainDDQN c go 
    }

(*
let go = ref true
runTraining go |> Async.Start

go.Value <- false
*)


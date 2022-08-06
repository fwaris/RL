#load "packages.fsx"
open AirSimCar
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
let lossFn = torch.nn.functional.smooth_l1_loss()
let modelFile = @"e:\s\ddqn\ddq_airsim.bin"

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
            do! clnt.reset()
            do! clnt.setCarControls({CarControls.Default with throttle = 1.0})
        else
            return failwith "unable to put car in api mode"
        do! Async.Sleep 1500
    }

let trainDDQN (clnt:CarClient) (go:bool ref) =
    resetCar clnt |> Async.AwaitTask |> Async.RunSynchronously
    let initState = CarEnvironment.RLState.Default
    let initCtrls = {CarControls.Default with throttle = 1.0}
    let burnIn = 32
    let learnEvery = 3
    let syncEvery = 100
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
                let! (state,ctrls,reward,isDone) = CarEnvironment.step clnt (state,ctrls) action 100 |> Async.AwaitTask
                printfn $"reward: {reward}, isDone: {isDone}, {action}, s,b,t = {ctrls.steering},{ctrls.brake},{ctrls.throttle}, speed={state.Speed}"

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
                        let td_est = DDQN.td_estimate states actions ddqn        
                        let td_est_d = td_est.data<float32>().ToArray() //ddqn invocations
                        let td_tgt = DDQN.td_target rewards nextStates dones ddqn
                        let loss = updateQ td_est td_tgt                                                          //update online model 
                        printfn $"{count}, loss: {loss}"

                    //periodically sync target model with online model
                    if count > syncEvery && count % syncEvery = 0 then 
                        ddqn.Model.Online.Module.save(modelFile) |> ignore
                        DDQNModel.sync ddqn.Model
                        printfn $"Exploration rate: {ddqn.Step.ExplorationRate}"

                    let count = count + 1
                    match isDone with CarEnvironment.NotDone -> () | _ ->  do! resetCar clnt |> Async.AwaitTask
                    
                    return! loop count state ctrls ddqn experienceBuff
                    
            with ex -> printfn "%A" ex.Message
        }
    loop 0 initState initCtrls initDDQN initExperience


let runTraining go =
    async {
        use c = new CarClient(AirSimCar.Defaults.options)
        c.Connect(AirSimCar.Defaults.address,AirSimCar.Defaults.port)       
        do! trainDDQN c go 
    }

(*
let go = ref true
runTraining go |> Async.Start

go.Value <- false
*)

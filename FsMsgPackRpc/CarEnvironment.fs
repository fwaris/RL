///RL 'environment' for training with the AirSim car simulator
///The agent can perform actions, compute rewards and observe state
module CarEnvironment
open System
open AirSimCar
open TorchSharp

let discreteActions = 6

///state we need for reinforcement learning
type RLState =
    {
        Pose                : torch.Tensor
        Speed               : float
        Collision           : bool
        DepthImage          : torch.Tensor       //depth perspective from front camera transformed
        PrevDepthImage      : torch.Tensor
    }
    with 
        static member Default =
            {
                Pose                = torch.zeros([|3L|],dtype=torch.float)
                DepthImage          = torch.zeros(1,84,84,dtype=torch.float)
                PrevDepthImage      = torch.zeros(1,84,84,dtype=torch.float)
                Speed               = 3.0
                Collision           = false
            }

type DoneReason = LowReward | Collision | Stuck | NotDone

///send the given action to car after translating it to the appropriate control message
let doAction (c:CarClient) action carCtrl = 
    let carCtrl = {carCtrl with throttle = 1.0; brake = 0.0}
    async {
        let ctl =
            match action with
            | 0 -> {carCtrl with throttle = 0.0; brake = 1.0}
            | 1 -> {carCtrl with steering = 0.0}
            | 2 -> {carCtrl with steering = 0.5}
            | 3 -> {carCtrl with steering = -0.5}
            | 4 -> {carCtrl with steering = 0.25}
            | _ -> {carCtrl with steering = -0.25}
        do! c.setCarControls(ctl) |> Async.AwaitTask
        do! Async.Sleep 1000
        return ctl
    }

///cached image request type
let imageRequest : ImageRequest[] = 
    [|
        {
            camera_name = "0"                           //get depth perspective from front camera
            image_type = ImageType.DepthPerspective
            pixels_as_float = true
            compress = false
        }
    |]

let transformImage (resp:ImageResponse) =
    let t1 = torch.tensor resp.image_data_float
    let t2 = 255.f.ToScalar() / torch.maximum(torch.ones_like t1, t1)
    let t3 = t2.reshape(1,resp.height,resp.width)
    torchvision.transforms.Resize(84,84).forward(t3)

///compute next state from previous state and 
///new observations from the environment
let getObservations (c:CarClient) prevState =
    task {
        let! images = c.simGetImages(imageRequest)
        let img = transformImage( images.[0])
        let! carState = c.getCarState()
        let! collInfo = c.simGetCollisionInfo()
        let nextState =
            {
                Speed           = carState.speed
                Collision       = collInfo.has_collided
                Pose            = carState.kinematics_estimated.position.ToArray() |> torch.tensor
                DepthImage      = img
                PrevDepthImage  = prevState.DepthImage
            }
        return nextState
    }

let computeReward (state:RLState) (ctrls:CarControls) =
    let MAX_SPEED = 300.
    let MIN_SPEED = 10.
    let THRESH_DIST = 3.5
    let BETA = 3.
    let pts =
        [
            (0, -1); (130, -1); (130, 125); (0, 125);
            (0, -1); (130, -1); (130, -128); (0, -128);
            (0, -1);        
        ]
        |> List.map (fun (x,y) -> torch.tensor([|float x;float y; 0.0|],dtype=torch.float))

    let car_pt = state.Pose
    let dist =         
        (10_000_000., List.pairwise pts)
        ||> List.fold (fun st (a,b) -> 
            let nrm = torch.linalg.cross(car_pt - a, car_pt - b).norm().ToDouble()
            let denom = (a - b).norm().ToDouble()
            let dist' = nrm/denom
            min st dist')
    let reward =
        if dist > THRESH_DIST then
            -3.0
        else    
            let reward_dist = Math.Exp(-BETA * dist) - 0.5
            let reward_speed = (((state.Speed - MIN_SPEED)/(MAX_SPEED - MIN_SPEED)) - 0.5)
            reward_dist + reward_speed
    let isDone =
        //printfn $"d:{dist},r:{reward},b:{ctrls.brake},s:{state.Speed},c:{state.Collision}"
        match reward, ctrls.brake, state.Speed, state.Collision with
        | rwrd,_,_,_ when rwrd < -1.0           -> LowReward //prob off course
        | _,br,sp,_ when br = 0.0 && sp < 1.0   -> Stuck
        | _,_,_,true                            -> Collision
        | _                                     -> NotDone
    reward,isDone

let step c (state,ctrls) action =
    task{  
        let! ctrls' = doAction c action ctrls
        let! state' = getObservations c state
        let reward,isDone = computeReward state' ctrls'
        return (state',ctrls',reward,isDone)
    }

let initCar (c:CarClient) = 
    task {
        let! _ = c.enableApiControl(true) 
        let! isApi = c.isApiControlEnabled() 
        if isApi then
            let! _ = c.armDisarm(true) 
            do! c.reset()
        else
            return failwith "unable to put car in api mode"
        do! Async.Sleep 100
    }

///an agent that uses random actions in the car environment.
///meant for testing the connectivity to AirSim
let startRandomAgent (go:bool ref) =
    let c = new CarClient(AirSimCar.Defaults.options)
    c.Connect(AirSimCar.Defaults.address,AirSimCar.Defaults.port)
    initCar c |> Async.AwaitTask |> Async.RunSynchronously
    let initState = RLState.Default
    let initCtrls = {CarControls.Default with throttle = 1.0}
    let initAction = 1
    let rng = Random()
    let rec loop state ctrls nextAction =
        async {
            try
                let! (state',ctrls',reward,isDone) = step c (state,ctrls) nextAction |> Async.AwaitTask
                printfn $"reward: {reward}, isDone: {isDone}"
                if not go.Value then
                    //do! initCar c |> Async.AwaitTask
                    c.Disconnect()
                    printfn "stopped"
                else
                    match isDone with
                        | NotDone -> 
                            let action = rng.Next(discreteActions)
                            return! loop state' ctrls' action
                        | doneRsn ->
                            printfn $"Done: {doneRsn}"                        
                            do! c.reset() |> Async.AwaitTask
                            do! Async.Sleep 1000 // need to wait for the car to settle down after reset
                            return! loop initState initCtrls initAction
            with ex -> printfn "%A" ex.Message
        }
    loop initState initCtrls initAction

(*
let go = ref true
startRandomAgent go |> Async.Start

go.Value <- false
*)


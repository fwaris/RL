#load "packages.fsx"
open System
open AirSimCar
open TorchSharp

///state we need for reinforcement learning
type RLState =
    {
        Position     : torch.Tensor
        PrevPosition : torch.Tensor
        PrevPose     : KinematicsState option
        Pose         : KinematicsState option
        Speed        : float
        Collision    : bool
    }
    with 
        static member Default =
            {
                Position     = torch.zeros([|3L|],dtype=torch.float)
                PrevPosition = torch.zeros([|3L|],dtype=torch.float)
                PrevPose     = None 
                Pose         = None 
                Speed        = 3.0
                Collision    = false
            }

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
        return ctl
    }

///send random actions to car (for testing)
let randomAgent (c:CarClient) (go:bool ref) =
    let state = ref {CarControls.Default with throttle = 0.2}
    let rng = Random()
    async {
        while go.Value do
            do! Async.Sleep 1000

            let action = rng.Next(6)           
            let! st' = doAction c action state.Value
            state.Value <- st'
    }

(*test with random agent 
let c1 = new CarClient(AirSimCar.Defaults.options)
c1.Connect(AirSimCar.Defaults.address,AirSimCar.Defaults.port)
c1.enableApiControl(true) |> Async.AwaitTask |> Async.RunSynchronously
c1.isApiControlEnabled() |> Async.AwaitTask |> Async.RunSynchronously
c1.armDisarm(true) |> Async.AwaitTask |> Async.RunSynchronously
c1.reset() |> Async.AwaitTask |> Async.RunSynchronously

let go = ref true
randomAgent c1 go |> Async.Start

go.Value <- false
c1.Disconnect()
*)

///cached image request type
let imageRequest : ImageRequest[] = 
    [|
        {
            camera_name = "0"
            image_type = ImageType.DepthPerspective
            pixels_as_float = true
            compress = false
        }
    |]

///compute next state from previous state and 
///new observations from the environment
let getObservations (c:CarClient) prevState =
    task {
        let! images = c.simGetImages(imageRequest)
        let! carState = c.getCarState()
        let! collInfo = c.simGetCollisionInfo()
        let nextState =
            {
                PrevPose = prevState.Pose
                Pose = Some carState.kinematics_estimated
                Speed = carState.speed
                Collision = collInfo.has_collided
                PrevPosition = prevState.Position
                Position = carState.kinematics_estimated.position.ToArray() |> torch.tensor
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

    match state.Pose with
    | Some pose ->
        let car_pt = torch.tensor(pose.position.ToArray())
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
            printfn $"r:{reward},b:{ctrls.brake},s:{state.Speed},c:{state.Collision}"
            match reward, ctrls.brake, state.Speed, state.Collision with
            | rwrd,_,_,_ when rwrd < -1.0           -> true      //significant negative reward
            | _,br,sp,_ when br = 0.0 && sp < 1.0   -> true      //car stuck
            | _,_,_,true                            -> true      //collision
            | _                                     -> false
        reward,isDone
    | None -> 
        1.0,false

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
    }

let start (go:bool ref) =
    let c = new CarClient(AirSimCar.Defaults.options)
    c.Connect(AirSimCar.Defaults.address,AirSimCar.Defaults.port)
    initCar c |> Async.AwaitTask |> Async.RunSynchronously
    let initState = RLState.Default
    let initCtrls = {CarControls.Default with throttle = 1.0}
    let initAction = 1
    let rng = Random()
    let rec loop state ctrls nextAction =
        async {
            let! (state',ctrls',reward,isDone) = step c (state,ctrls) nextAction |> Async.AwaitTask
            printfn $"reward: {reward}, isDone: {isDone}"
            if not go.Value then
                //do! initCar c |> Async.AwaitTask
                c.Disconnect()
                printfn "stopped"
            else
                if isDone then
                    do! c.reset() |> Async.AwaitTask
                    return! loop initState initCtrls initAction
                else
                    let action = rng.Next(6)
                    return! loop state' ctrls' action
        }
    loop initState initCtrls initAction

(*
let go = ref true
start go |> Async.Start

go.Value <- false
*)




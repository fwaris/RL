#load "packages.fsx"
open System
open AirSimCar
open TorchSharp

type RLState =
    {
        Position     : torch.Tensor
        PrevPosition : torch.Tensor
        PrevPose     : KinematicsState
        Pose         : KinematicsState
        Collision    : bool
    }

let do_action (c:CarClient) action carCtrl = 
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

let random_agent (c:CarClient) (go:bool ref) =
    let state = ref {CarControls.Default with throttle = 0.2}
    let rng = Random()
    async {
        while go.Value do
            do! Async.Sleep 1000

            let action = rng.Next(6)           
            let! st' = do_action c action state.Value
            state.Value <- st'
    }

let compute_reward (state:RLState) =
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

    let car_pt = torch.tensor(state.Pose.position.ToArray(),dtype=torch.float)
    let dist =         
        (10_000_000.f.ToScalar(),pts |> List.pairwise)
        ||> List.fold (fun st (a,b) -> 
            let nrm = torch.linalg.cross(car_pt - a, car_pt - b).norm()
            let denom = (a - b).norm()
            let dist' = nrm/denom
            min st dist')
    let dist = dist.ToDouble()
    let reward =
        if dist > THRESH_DIST then
            -3.0
        else    
            let reward_dist = Math.Exp(-BETA * dict) - 0.5





(*
    pts = [
                np.array([x, y, 0])
                for x, y in [

                ]
            ]
        car_pt = self.state["pose"].position.to_numpy_array()

        dist = 10000000
        for i in range(0, len(pts) - 1):
            dist = min(
                dist,
                np.linalg.norm(
                    np.cross((car_pt - pts[i]), (car_pt - pts[i + 1]))
                )
                / np.linalg.norm(pts[i] - pts[i + 1]),
            )

        # print(dist)
        if dist > THRESH_DIST:
            reward = -3
        else:
            reward_dist = math.exp(-BETA * dist) - 0.5
            reward_speed = (
                (self.car_state.speed - MIN_SPEED) / (MAX_SPEED - MIN_SPEED)
            ) - 0.5
            reward = reward_dist + reward_speed

        done = 0
        if reward < -1:
            done = 1
        if self.car_controls.brake == 0:
            if self.car_state.speed <= 1:
                done = 1
        if self.state["collision"]:
            done = 1

        return reward, done
    
    (*
    *)


let c1 = new CarClient(AirSimCar.Defaults.options)
c1.Connect(AirSimCar.Defaults.address,AirSimCar.Defaults.port)
c1.enableApiControl(true) |> Async.AwaitTask |> Async.RunSynchronously
c1.isApiControlEnabled() |> Async.AwaitTask |> Async.RunSynchronously
c1.armDisarm(true) |> Async.AwaitTask |> Async.RunSynchronously
c1.reset() |> Async.AwaitTask |> Async.RunSynchronously

let go = ref true
random_agent c1 go |> Async.Start

go.Value <- false
c1.Disconnect()


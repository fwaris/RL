///RL 'environment' for training with the AirSim car simulator
///The agent can perform actions, compute rewards and observe state
module CarEnvironment
open System
open AirSimCar
open TorchSharp
open System.Numerics
open FSharp.Control

let discreteActions = 6
let IMG_DIM=256
let MAX_EPISODE_STEPS = 30
let DTYPE = torch.float32

type Seg = {p1:Vector3; p2:Vector3}

type LogLevel = Verbose | Quite with member this.isVerbose() = match this with Verbose -> true | _ -> false

type DoneReason = OffCourse | Collision | Stuck | NotDone | EpisodeEnd

type CarRLState = {
    CarId               : string
    DoneReason          : DoneReason
    Reward              : float
    Controls            : CarControls
    Position            : Vector3
    Orientation         : Quaternion
    Speed               : float
    Velocity            : Vector3
    Collision           : CollisionInfo
    SegImage            : torch.Tensor
    DepthImage          : torch.Tensor       //depth perspective from front camera transformed
    SceneImage          : torch.Tensor
    PrevSegImage        : torch.Tensor
    PrevDepthImage      : torch.Tensor
    PrevSceneImage      : torch.Tensor
    CombinedImage       : torch.Tensor
    PrevCombinedImage   : torch.Tensor
    WasReset            : bool
    PrevSpeeds          : float list
    ActionHistory       : int list
    ControlsHistory     : CarControls list
}
    with 
        static member Create carId =
            {
                CarId               = carId
                DoneReason          = NotDone
                Reward              = 0.0
                Controls            = CarControls.Default
                Position            = Vector3()
                Orientation         = Quaternion()
                SegImage            = torch.zeros(1,IMG_DIM,IMG_DIM,dtype=DTYPE)
                DepthImage          = torch.zeros(1,IMG_DIM,IMG_DIM,dtype=DTYPE)
                PrevDepthImage      = torch.zeros(1,IMG_DIM,IMG_DIM,dtype=DTYPE)
                SceneImage          = torch.zeros(1,IMG_DIM,IMG_DIM,dtype=DTYPE)
                PrevSegImage        = torch.zeros(1,IMG_DIM,IMG_DIM,dtype=DTYPE)      
                PrevSceneImage      = torch.zeros(1,IMG_DIM,IMG_DIM,dtype=DTYPE)      
                CombinedImage       = torch.zeros(1,4,IMG_DIM,IMG_DIM,dtype=DTYPE)      
                PrevCombinedImage   = torch.zeros(1,4,IMG_DIM,IMG_DIM,dtype=DTYPE)      
                Speed               = 0.0
                Velocity            = Vector3()
                Collision           = CollisionInfo.Default
                WasReset            = false
                PrevSpeeds          = []
                ActionHistory       = []
                ControlsHistory     = []
            }

///state we need for reinforcement learning
type RLState =
    {
        CarStates           : CarRLState list
        EpisodeSteps        : int
    }
    with 
        static member Create carIds = {CarStates = carIds |> List.map CarRLState.Create; EpisodeSteps = 0 }


module Road =
    module Markers = 
        let ox = 0.f
        let oy = -1.f
        let northMostX = 127.7f
        let northInterX = 80.f
        let northLeastX = -127.5f
        let eastMostY = 126.3f
        let eastLeastY = -128.8f

    module Vertices = 
        open Markers
        //vertices
        let v00 = Vector3(ox,oy,0.f)
        let vne = Vector3(northMostX,eastMostY,0.f)
        let v0e = Vector3(ox,eastMostY,0.f)
        let vse = Vector3(northLeastX,eastMostY,0.f)
        let vs0 = Vector3(northLeastX,oy,0.f)
        let vsw = Vector3(northLeastX,eastLeastY,0.f)
        let v0w = Vector3(ox,eastLeastY,0.f)
        let viw = Vector3(northInterX,eastLeastY,0.f)
        let vnw = Vector3(northMostX,eastLeastY,0.f)
        let vn0 = Vector3(northMostX,oy,0.f)
        let vi0 = Vector3(northInterX,oy,0.f)
        let all = [v00;vne;v0e;vse;vs0;vsw;v0w;viw;vnw;vn0;vi0]

    open Vertices
    let network = 
        //edges
        [
            {p1=v00; p2 = vi0}
            {p1=v00; p2 = v0e}
            {p1=v00; p2 = vs0}
            {p1=v00; p2 = v0w}
            {p1=viw; p2 = vnw}
            {p1=viw; p2 = vi0}
            {p1=viw; p2 = v0w}
            {p1=vn0; p2=vnw}
            {p1=vn0; p2=vi0}
            {p1=vn0; p2=vne}
            {p1=vne; p2=v0e}
            {p1=vse; p2=v0e}
            {p1=vse; p2=vs0}
            {p1=vsw; p2=v0w}
            {p1=vsw; p2=vs0}
        ]

let enableApiControl carIds (clnt:CarClient) = 
    carIds
    |> AsyncSeq.ofSeq 
    |> AsyncSeq.iterAsync(fun carId -> async{
        let! _ = clnt.enableApiControl(true,vehicle_name=carId) |> Async.AwaitTask
        let! isApi = clnt.isApiControlEnabled(vehicle_name=carId) |> Async.AwaitTask
        if isApi then
            let! _ = clnt.armDisarm(true,vehicle_name=carId)  |> Async.AwaitTask
            do!  clnt.reset() |> Async.AwaitTask
        else
            return failwith "unable to put car in api mode"        
        do! Async.Sleep 10
    })


///send the given action to car after translating it to the appropriate control message
let doAction (c:CarClient) carId action carCtrl = 
    let carCtrl = {carCtrl with throttle = 0.50; brake = 0.0}
    async {
        try
            let ctl =
                match action with
                | 0 -> {carCtrl with throttle = 0.0; brake = 1.0}
                | 1 -> {carCtrl with steering = 0.0}
                | 2 -> {carCtrl with steering = 0.5}
                | 3 -> {carCtrl with steering = -0.5}
                | 4 -> {carCtrl with steering = 0.25}
                | _ -> {carCtrl with steering = -0.25}
            do! c.setCarControls(ctl,vehicle_name=carId) |> Async.AwaitTask
            return ctl
        with ex ->
            printfn $"{ex.Message},{ex.StackTrace}"
            return carCtrl
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
        {
            camera_name = "0"                           //get segmented image from front camera
            image_type = ImageType.Segmentation
            pixels_as_float = true
            compress = false
        }
        {
            camera_name = "0"                           //get scene from front camera
            image_type = ImageType.Scene
            pixels_as_float = false
            compress = false
        }
    |]

//scale camera image and resize 
let transformSceneImage (resp:ImageResponse) =
    use t1 = torch.tensor(resp.image_data_uint8,dtype=DTYPE)
    use t3 = t1.reshape(resp.height,resp.width,1)
    use t4 = t3.permute(2,0,1)                                         //HWC -> CHW 
    torchvision.transforms.Resize(IMG_DIM,IMG_DIM).call(t4)
let transformSegImage (resp:ImageResponse) =
    use t1 = torch.tensor(resp.image_data_float,dtype=DTYPE)
    use t3 = t1.reshape(resp.height,resp.width,1)
    use t4 = t3.permute(2,0,1)                                         //HWC -> CHW 
    torchvision.transforms.Resize(IMG_DIM,IMG_DIM).call(t4)
let transformDepthImage (resp:ImageResponse) =
    use t1 = torch.tensor(resp.image_data_float,dtype=DTYPE)
    use t2 = 255.f.ToScalar() / torch.maximum(torch.ones_like t1, t1)  //inverts distance near is higher; max 'nearness' is 255
    use t3 = t2.reshape(resp.height,resp.width,1)
    use t4 = t3.permute(2,0,1)                                         //HWC -> CHW 
    torchvision.transforms.Resize(IMG_DIM,IMG_DIM).call(t4)

///compute next state from previous state and 
///new observations from the environment
let getObservations (c:CarClient) prevState =
    task {
        try
            let carId = prevState.CarId
            let! carState = c.getCarState(vehicle_name=carId)
            let! gtKinematics = c.simGetGroundTruthKinematics(vehicle_name=carId)
            let! images = c.simGetImages(imageRequest, vehicle_name=carId)
            let! collInfo = c.simGetCollisionInfo(vehicle_name=carId)
            let dimg = try transformDepthImage( images.[0]) with _ -> prevState.DepthImage //occaisionally a bad image is received
            let sgimg = try transformDepthImage( images.[1]) with _ -> prevState.SegImage //occaisionally a bad image is received
            let simg = try transformSegImage( images.[2]) with _ -> prevState.SceneImage //occaisionally a bad image is received
            let combined = torch.stack ([dimg; prevState.DepthImage; sgimg; simg], dim=1)
            let combined = combined.clone().contiguous()            
            let p = gtKinematics.position
            let q = gtKinematics.orientation
            let position = p.ToVector3()
            let orientation = q.ToQuaternion()
            let nextState =
                { prevState with
                    Speed             = carState.speed
                    Velocity          = gtKinematics.linear_velocity.ToVector3()
                    Collision         = collInfo
                    Position          = position
                    Orientation       = orientation
                    SegImage          = sgimg
                    PrevSegImage      = prevState.SegImage
                    DepthImage        = dimg
                    PrevDepthImage    = prevState.DepthImage
                    SceneImage        = simg
                    PrevSceneImage    = prevState.SceneImage
                    CombinedImage     = combined
                    PrevCombinedImage = prevState.CombinedImage
                    PrevSpeeds        = carState.speed::prevState.PrevSpeeds |> List.truncate 5
                }
            return nextState
        with ex ->
            printfn $"{ex.Message},{ex.StackTrace}"
            return prevState
    }


let MAX_SPEED = 5.
let MIN_SPEED = 0.4
let THRESH_DIST = 6.0f
let BETA = 3.


let distToSeg (car_pt: Vector3) (seg: Seg) =
    let a = seg.p1
    let b = seg.p2
    let ab = b - a
    let ap = car_pt - a

    let ab_len2 = Vector3.Dot(ab, ab)
    
    // Handle degenerate segment (p1 == p2)
    let t =
        if ab_len2 < 1e-8f then 
            0.0f 
        else 
            Vector3.Dot(ap, ab) / ab_len2

    // Clamp to the segment
    let t_clamped = 
        if t < 0.0f then 0.0f
        elif t > 1.0f then 1.0f
        else t

    // Closest point on the segment
    let closest = a + t_clamped * ab

    // Distance between point and closest point
    Vector3.Distance(car_pt, closest)


let closestRoadSeg (car_pt:Vector3) (roadSements: Seg list) =
    let seg,dist =         
        ((None,10_000_000.f), roadSements)
        ||> List.fold (fun (s,st) seg  ->
            let dist = distToSeg car_pt seg
            if dist < st then Some seg, dist else s,st)
    seg.Value,dist

let rotate (v:Vector3, q:Quaternion) = 
    let vQuat = Quaternion(v,0.f)
    let rotated = q * vQuat * Quaternion.Conjugate(q)
    Vector3(rotated.X, rotated.Y, rotated.Z)

let movingTowardsP2 (position:Vector3) (orientation:Quaternion) (seg:Seg) = 
    let forward = Vector3.UnitX
    let fworld = rotate (forward,orientation)
    let toP1 = Vector3.Normalize(seg.p1 - position)
    let toP2 = Vector3.Normalize(seg.p2 - position)
    let dotP1 = Vector3.Dot(toP1, fworld)
    let dotP2 = Vector3.Dot(toP2, fworld)
    dotP2 > dotP1

let isRightOfLine (movingTowardsP2:bool) (seg:Seg) (car:Vector3) = 
    let a,b = if movingTowardsP2 then seg.p1,seg.p2 else seg.p2,seg.p1
    //2d vectors assume Z-axis=0
    let ab = b - a
    let ac = car - a
    let tval = ab.X * ac.Y - ab.Y * ac.X //cross2D
    tval < 0.0f

let printCarState (state:CarRLState) (towardsP2:bool) (rightOfLine:bool) (seg:Seg) (dist:float32) = 
    let tBack,tFront = if towardsP2 then seg.p1,seg.p2 else seg.p2,seg.p1
    let c = state.Position
    let x1,y1,x2,y2 = tBack.X, tBack.Y, tFront.X,tFront.Y
    printfn $"[%0.1f{x1}, %0.1f{y1} -> %0.1f{x2}, %0.1f{y2}], {state.CarId} %0.1f{c.X}, %0.1f{c.Y}, right:{rightOfLine}, dist:%02.1f{dist}"

let computeReward (logLevel:LogLevel ref) (state:CarRLState) =
    let mutable car_pt = state.Position
    car_pt.Z <- 0.f
    let seg,dist = closestRoadSeg car_pt Road.network // (rewardSegs.Value)
    let towardsP2 = movingTowardsP2 state.Position state.Orientation seg
    let rightOfLine = isRightOfLine towardsP2 seg car_pt

    //find distance to center line of the nearest road
    if logLevel.Value.isVerbose() then
        printCarState state towardsP2 rightOfLine seg dist
    let threshold_dist = if state.WasReset then THRESH_DIST + 2.0f else THRESH_DIST //car position is not stable its reset
    let reward,doneReason =
        if state.Collision.has_collided then 
            let penality  = 
                if state.Collision.object_name = "Car1" || state.Collision.object_name="Car2" then 
                    let orientation = state.Collision.normal.ToVector3()
                    let isFrontal = CollisionUtil.isCollisionFacingFrontByNormal state.Velocity state.Orientation  orientation 0.5f 0.2f
                    if isFrontal then 
                        -10.0
                    else
                        -3.0
                else
                    -10.0
            penality,Collision
        elif dist > threshold_dist then
            printfn $"Off Course: {state.CarId}: {car_pt}, seg: {seg}, dist:{dist}"
            -5.0,OffCourse
        elif state.PrevSpeeds.Length >= 5 && state.PrevSpeeds |> List.forall (fun x -> x < 0.01) then 
            -3.0,Stuck
        else
            let reward_dist = Math.Exp(-BETA * float dist) - 0.5 
            let reward_right_side = if rightOfLine then reward_dist + 0.5 else reward_dist - 0.5
            let reward_speed = (((state.Speed - MIN_SPEED)/(MAX_SPEED - MIN_SPEED)) - 0.5)
            let reward = reward_right_side + reward_speed
            reward,NotDone
    {state with WasReset=false; Reward=reward; DoneReason=doneReason}

let resetCarForNextEpisode s = 
    {s with 
        WasReset        = true
        PrevSpeeds      = []
        ActionHistory   = []
        ControlsHistory = []
    }

let step logLevel c (state:RLState) (actions:int list) =
    async{  

        //take action
        let! carStates = 
            List.zip state.CarStates actions 
            |> AsyncSeq.ofSeq 
            |> AsyncSeq.mapAsync(fun (st,a) -> async {
                let! ctrls = doAction c st.CarId a st.Controls
                return {st with Controls=ctrls; ActionHistory = a::st.ActionHistory; ControlsHistory=ctrls::st.ControlsHistory}
            })
            |> AsyncSeq.toListAsync

        // allow some time for the actions to take affect before observing
        do! Async.Sleep 700 

        //observe and compute reward
        let! carStates = 
            carStates 
            |> AsyncSeq.ofSeq
            |> AsyncSeq.mapAsync (fun carState -> async {
                let! carState = getObservations c carState |> Async.AwaitTask
                let carState = computeReward logLevel carState
                let doneReason = match carState.DoneReason with NotDone when state.EpisodeSteps > MAX_EPISODE_STEPS -> EpisodeEnd | _ -> carState.DoneReason
                return {carState with DoneReason=doneReason}
            })
            |> AsyncSeq.toListAsync

        let isSomeCarDone =  carStates |> List.exists(fun s -> s.DoneReason <> NotDone) //then entire episode is done
        let episodeSteps = if isSomeCarDone then 0 else state.EpisodeSteps+1  
        let carStates = if episodeSteps = 0 then carStates |> List.map resetCarForNextEpisode else carStates
       
        return {state with CarStates = carStates; EpisodeSteps=episodeSteps}
    }

let rng = Random()
let getDistinctRandomNumbers (n: int) (max: int) =
    if n > max then
        invalidArg "n" "n must be less than or equal to max"

    let numbers = [| 0 .. max - 1 |]

    // Fisher-Yates shuffle
    for i in max - 1 .. -1 .. 1 do
        let j = rng.Next(i + 1)
        let temp = numbers.[i]
        numbers.[i] <- numbers.[j]
        numbers.[j] <- temp

    numbers.[0 .. n - 1] |> Array.toList

let APPROACH_DIST = 30.f

let approachCorner(tStart:Vector3,tEnd:Vector3) =
    let segLen = (tEnd - tStart).Length()
    let segFrac = 1.0f - APPROACH_DIST / segLen
    let intrpltd = (1.0f - segFrac) * tStart + segFrac * tEnd
    intrpltd 

let approachSameRandomCorner() = 
    let seg = Road.network.[rng.Next(Road.network.Length)]
    let _,front = if rng.NextDouble() < 0.5 then seg.p1, seg.p2 else seg.p2,seg.p1
    let segs = Road.network |> List.filter (fun s -> s.p1 = front || s.p2 = front) |> List.take 2
    let s1,s2 = segs.[0], segs.[1]
    let back1 = if s1.p1 = front then s1.p2 else s1.p1
    let back2 = if s2.p1 = front then s2.p2 else s2.p1
    let interpld1 = approachCorner(back1,front)
    let interpld2 = approachCorner(back2,front)
    [(back1,front),interpld1; (back2,front),interpld2]    

let randRoadPointsApproachCorner n = 
    if n = 2 then 
        approachSameRandomCorner()
    else
        let ns = getDistinctRandomNumbers n Road.network.Length
        ns 
        |> List.map(fun i -> 
            let seg = Road.network.[i]
            let back,front = if rng.NextDouble() < 0.5 then seg.p1, seg.p2 else seg.p2,seg.p1
            (back,front),approachCorner (back,front))

module Poses =
    let front = "f", {x_val=0.0; y_val=0.0; z_val=0.0; w_val=1.0}
    let left  = "l", {x_val=0.0; y_val=0.0; z_val=0.7071; w_val=0.7071}   // 90° CCW
    let back  = "b", {x_val=0.0; y_val=0.0; z_val=1.0;    w_val=0.0}      // 180°
    let right = "r", {x_val=0.0; y_val=0.0; z_val= -0.7071; w_val=0.7071}  // 90° CW
    let all   = [front; right; back; left]

open System.Numerics

let lookAt (car: Vector3) (target: Vector3) =
    let forward = Vector3.UnitX
    let dir = Vector3.Normalize(target - car)

    // Handle degenerate ‘no movement’ case
    if dir.LengthSquared() < 1e-8f then
        Quaternion.Identity
    else
        let dot = Vector3.Dot(forward, dir)

        // If vectors are opposite → pick any perpendicular axis
        if dot < -0.999999f then
            // rotate 180° around Y (or Z)
            Quaternion.CreateFromAxisAngle(-Vector3.UnitZ, System.MathF.PI)
        else
            let axis = Vector3.Cross(forward, dir)
            let w = 1.0f + dot
            let q = Quaternion(axis.X, axis.Y, axis.Z, w)
            Quaternion.Normalize q

let getPose ((tStart:Vector3,tEnd:Vector3),(tCar:Vector3)) = 
    let q = lookAt tCar tEnd
    {
        position = Vector3r.FromVector3 tCar
        orientation = Quaternionr.FromQuaternion q
    }    

let homePath = lazy(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile))
let saveImage (clnt:CarClient) carId = async {
    let! image = clnt.simGetImage("front_right",ImageType.Scene ,vehicle_name=carId) |> Async.AwaitTask
    if image <> null then
        let path = IO.Path.Combine(homePath.Value,$"{carId}.png")
        IO.File.WriteAllBytes(path, image)
    ()
}

let MAX_POSE_RESET_TRIES = 1 //3
let rec confirmSetPose count (clnt:CarClient) (carId:string) (pose:Pose) = async {
    do! clnt.simSetObjectPose(carId,pose,true) |> Async.AwaitTask |> Async.Ignore
    let! pose' = clnt.simGetObjectPose(carId) |> Async.AwaitTask
    let dx,dy,dz = abs (pose.position.x_val - pose'.position.x_val), abs(pose.position.y_val - pose'.position.y_val), abs(pose.position.z_val - pose'.position.z_val)
    if (dx > 0.1 || dy > 0.1) && count < MAX_POSE_RESET_TRIES then         
        do! Async.Sleep 200
        printfn $"recalibrating pose dx:%0.1f{dx}, dy:%0.1f{dy}, dz:%0.1f{dz}"
        return! confirmSetPose (count + 1) clnt carId pose
    else
        return()
}

let reset1Car (clnt:CarClient) (wait:int) (((vFrm,vTo),poseV3),cst) =  async {
    do! clnt.setCarControls({CarControls.Default with throttle = 0.0; brake=1.0},vehicle_name=cst.CarId) |> Async.AwaitTask    
    let pose = getPose ((vFrm,vTo),poseV3)
    do! confirmSetPose 0 clnt cst.CarId pose
    do! Async.Sleep wait
    let seg,dist = closestRoadSeg (pose.position.ToVector3()) Road.network
    printfn $"{cst.CarId} pos:[%0.1f{pose.position.x_val},%0.1f{pose.position.y_val}], seg: [{vFrm},{vTo} dist: {dist}"
    //do! saveImage clnt cst.CarId                                                              
    return {cst with Position=poseV3; Speed=0}
}

let resetCars (clnt:CarClient) (carStates:CarRLState list) = async {
    try
            
        do! clnt.reset() |> Async.AwaitTask
        do! Async.Sleep 11
        let poses = randRoadPointsApproachCorner carStates.Length
        let! carStates =
            carStates
            |> List.zip poses
            |> AsyncSeq.ofSeq
            |> AsyncSeq.mapAsync (reset1Car clnt 0)
            |> AsyncSeq.toListAsync
        do! Async.Sleep 300
        return carStates
    with ex ->
        printfn $"resetCar error: {ex.Message}, {ex.StackTrace}"
        return carStates
}

let mutable _lastResetTime = DateTime.MinValue
let stepRun logLevel clnt (state:RLState) (actions:int list) =
    async{  

        //take action
        let! carStates = 
            List.zip state.CarStates actions 
            |> AsyncSeq.ofSeq 
            |> AsyncSeq.mapAsync(fun (st,a) -> async {
                let! ctrls = doAction clnt st.CarId a st.Controls
                return {st with Controls=ctrls; ActionHistory = a::st.ActionHistory; ControlsHistory=ctrls::st.ControlsHistory}
            })
            |> AsyncSeq.toListAsync

        // allow some time for the actions to take affect before observing
        do! Async.Sleep 700 

        //observe and determine car state
        let! carStates = 
            carStates 
            |> AsyncSeq.ofSeq
            |> AsyncSeq.mapAsync (fun carState -> async {
                let! carState = getObservations clnt carState |> Async.AwaitTask
                let carState = computeReward logLevel carState
                return carState
            })
            |> AsyncSeq.mapAsync (fun cst -> async {
                match cst.DoneReason with 
                NotDone -> return cst 
                | _     -> 
                            let tnow = DateTime.Now
                            let diff = tnow - _lastResetTime
                            if diff < TimeSpan.FromSeconds(1.) then 
                                printfn "Early reset"                              
                            _lastResetTime <- tnow
                            let cst = resetCarForNextEpisode cst
                            let ((vFrm,vTo),poseV3) = (randRoadPointsApproachCorner 1).[0]
                            return! reset1Car clnt 300 (((vFrm,vTo),poseV3),cst)
            })
            |> AsyncSeq.toListAsync
       
        return {state with CarStates = carStates}
    }


///an agent that uses random actions in the car environment.
///meant for testing the connectivity to AirSim
let startRandomAgent carId doLog (go:bool ref) =
    let c = new CarClient(AirSimCar.Defaults.options)
    c.Connect(AirSimCar.Defaults.address,AirSimCar.Defaults.port)
    enableApiControl [carId]  c |> Async.RunSynchronously    
    let initState = RLState.Create [carId]
    let initAction = 1
    let rng = Random()
    let rec loop state nextAction =
        async {
            try
                let! state = step doLog c state [nextAction]
                let cs = state.CarStates.[0]
                printfn $"reward: {cs.Reward}, isDone: {cs.DoneReason}"
                if not go.Value then
                    //do! initCar c |> Async.AwaitTask
                    c.Disconnect()
                    printfn "stopped"
                else
                    match cs.DoneReason with
                        | NotDone -> 
                            let action = rng.Next(discreteActions)
                            return! loop state action
                        | doneRsn ->
                            printfn $"Done: {doneRsn}"                        
                            do! c.reset() |> Async.AwaitTask
                            do!
                                randRoadPointsApproachCorner 1
                                |> List.zip [carId]
                                |> AsyncSeq.ofSeq
                                |> AsyncSeq.iterAsync (fun (carId,(seg,pose)) -> c.simSetObjectPose(carId,getPose (seg,pose),true) |> Async.AwaitTask |> Async.Ignore)
                            do! Async.Sleep 1000 // need to wait for the car to settle down after reset
                            return! loop initState initAction
            with ex -> printfn "%A" ex.Message
        }
    loop initState initAction

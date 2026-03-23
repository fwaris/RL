open AirSimCar
open FSharp.Control

let carIds = ["Car1"; "Car2"] //"PhysXCar"
//let carIds = ["PhysXCar"]

let resetCars (clnt:CarClient) = 
    async {
        try
            
            do! clnt.reset() |> Async.AwaitTask
            do! Async.Sleep 11
            do!
                CarEnvironment.randRoadPointsApproachCorner carIds.Length 
                |> List.zip carIds
                |> AsyncSeq.ofSeq
                |> AsyncSeq.iterAsync (fun (carId,((p1,p2),pose)) -> async {
                    printfn $"Corner {p1},{p2} - {carId}"
                    do! clnt.setCarControls({CarControls.Default with throttle = 0.1},vehicle_name=carId) |> Async.AwaitTask
                    do! clnt.simSetObjectPose(carId,CarEnvironment.getPose ((p1,p1),pose),true) |> Async.AwaitTask |> Async.Ignore
                })
        with ex ->
            printfn $"resetCar error: {ex.Message}, {ex.StackTrace}"
    }

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


let poseTest2() = 
    let pose = 
        { position = { x_val = 9.999998093
                       y_val = 126.3000031
                       z_val = 0.0 }
          orientation = { w_val = nan
                          x_val = nan
                          y_val = nan
                          z_val = nan } }
    let target = Vector3(0.f, 126.3f, 0.f)
    let car_pt = pose.position.ToVector3()
    let seg = {CarEnvironment.p1=Vector3(127.7f, 126.3f, 0.f); CarEnvironment.p2=Vector3(0.f, 126.3f, 0.f)}
    let pose2 = lookAt car_pt target
    let pose3 = CarEnvironment.lookAt car_pt target
    ()


let poseTest() = async {
    let c = new CarClient(AirSimCar.Defaults.options)
    c.Connect(AirSimCar.Defaults.address,AirSimCar.Defaults.port)      
    resetCars c |> Async.RunSynchronously
    while true do
        do! Async.Sleep 5000
        do! Async.Sleep 10
        let pts = CarEnvironment.randRoadPointsApproachCorner 1
        let target = pts.[0] |> fst |> snd
        let pose = pts.[0] |> CarEnvironment.getPose
        printfn $"{pose}"
        printfn $"{target}"
        do! c.simPlotPoints([Vector3r.FromVector3 target],duration=5.0) |> Async.AwaitTask
        let car_pt = pose.position.ToVector3()
        let seg,dist = CarEnvironment.closestRoadSeg car_pt CarEnvironment.Road.network // (rewardSegs.Value)
        printfn $"sec {seg.p1}<->{seg.p2}: dist {dist}"

        let! _ = c.simSetObjectPose(carIds.[0],pose,true) |> Async.AwaitTask
        ()
}

type MCommand = MReset | MGo
let ev1 : Event<MCommand> = Event<MCommand>()
let ev1P = ev1.Publish
let multiCarTest()  = 
    let c = new CarClient(AirSimCar.Defaults.options)
    c.Connect(AirSimCar.Defaults.address,AirSimCar.Defaults.port)      
    CarEnvironment.enableApiControl carIds c |> Async.RunSynchronously
    let assets = c.simListAssets().Result
    let objects = c.simListSceneObjects().Result
    printfn $"asssets: {assets.Length}, scene objects: {objects.Length}"
    let rec loop() = async {
        while true do
        match! Async.AwaitEvent ev1P with
        | MReset -> do! resetCars c
        | MGo    -> 
            do!
                carIds
                |> AsyncSeq.ofSeq
                |> AsyncSeq.iterAsync(fun cid -> async{
                    let ctrls = {CarControls.Default with throttle = 0.5}
                    do! c.setCarControls(ctrls, vehicle_name = cid) |> Async.AwaitTask
                })
        return! loop()
    }
    loop()


let cornerTest() = 
    let c = new CarClient(AirSimCar.Defaults.options)
    c.Connect(AirSimCar.Defaults.address,AirSimCar.Defaults.port)          
    let pts = CarEnvironment.Road.Vertices.all 
    let rec loop iPt iPose =  async {
        let t = pts.[iPt]
        let n,q = CarEnvironment.Poses.all.[iPose]
        let pose = 
            {
                position = 
                    {    
                        x_val = float t.X
                        y_val = float t.Y
                        z_val = 0.0
                    }

                orientation = q
            }    
        printfn  $"{t.X},{t.Y}, {n}"
        let! _ = c.simSetObjectPose(carIds.[0],pose,true) |> Async.AwaitTask        
        do! Async.Sleep 5000
        let iPose = (iPose + 1) % CarEnvironment.Poses.all.Length
        let iPt = if iPose = 0  then iPt + 1 else iPt
        let iPt = iPt % (pts.Length)
        return! loop iPt iPose
    }
    //let scenObs = c.simListSceneObjects().Result
    CarEnvironment.Road.network |> Seq.iter (fun x -> 
        let ps = [x.p1 |> Vector3r.FromVector3; x.p2 |> Vector3r.FromVector3]
        let r = c.simPlotLineList(ps,is_persistent=true).Result
        ())
    let strs = pts |> List.map (fun x-> $"{x}")
    let _ = c.simPlotStrings(strs, pts |> List.map Vector3r.FromVector3, scale=2.0).Result
    loop 0 0

let roadPoints() = 
    let c = new CarClient(AirSimCar.Defaults.options)
    c.Connect(AirSimCar.Defaults.address,AirSimCar.Defaults.port)          
    let pts = CarEnvironment.Road.network 
    //let pts = CarEnvironment._rewardSegs.Value |> List.distinct
    let t = pts.[3].p1
    let n,q = CarEnvironment.Poses.all.[0]
    let pose = 
        {
            position = 
                {    
                    x_val = float t.X
                    y_val = float t.Y
                    z_val = 0.0
                }

            orientation = q
        }    
    let _ =  c.simSetObjectPose(carIds.[0],pose,true).Result
    let rec loop() =  async {                
        do! Async.Sleep 1000
        let! pose = c.simGetObjectPose(carIds.[0]) |> Async.AwaitTask
        let p = pose.position
        printfn $"{p.x_val}, {p.y_val}"
        return! loop()
    }
    //let scenObs = c.simListSceneObjects().Result
    pts |> Seq.iter (fun p -> 

        let ps = [Vector3r.FromVector3 p.p1; Vector3r.FromVector3 p.p2]
        let r = c.simPlotLineList(ps,is_persistent=true).Result
        ())
    let positions = pts |> List.collect (fun x->[x.p1; x.p2]) |> List.distinct 
    let strs = positions |> List.map (fun x-> $"{x}")
    let _ = c.simPlotStrings(strs, positions |> List.map Vector3r.FromVector3, scale=2.0).Result
    loop()

let go = ref true
let logLevel = ref CarEnvironment.Verbose

poseTest() |> Async.Start //shows where is car randomly spawn 
//cornerTest() |> Async.Start   //shows car at each corner with four different orientations
//multiCarTest() |> Async.Start

let printUsage() = System.Console.WriteLine("x=quit; r=reset; g=go")

printUsage()
let rec loop() = 
    let k = System.Console.ReadKey()
    match k.KeyChar with 
    | 'x' -> go.Value <- false
    | 'r' -> ev1.Trigger(MReset)
             loop()
    | 'g' -> ev1.Trigger(MGo)
             loop()
    | _ -> printUsage(); loop()
loop()


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


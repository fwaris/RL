open System
open MessagePack
open MessagePack.FSharp
open MessagePack.Resolvers
open FsMsgPackRpc

let address = "localhost"
let port = 41451
let resolver = Resolvers.CompositeResolver.Create(
                FSharpResolver.Instance,
                StandardResolver.Instance)
let options = MessagePackSerializerOptions.Standard.WithResolver(resolver)

[<MessagePackObject(true)>]
type CarControls =
    {
        [<Key(0)>] throttle        : float32
        [<Key(1)>] steering        : float32 
        [<Key(2)>] brake           : float32
        [<Key(3)>] handbrake       : bool
        [<Key(4)>] is_manual_gear  : bool
        [<Key(5)>] manual_gear     : int
        [<Key(6)>] gear_immediate  : bool
    }
    with
        static member Default =
            {
                throttle        = 0.f
                steering        = 0.f
                brake           = 0.f
                handbrake       = false
                is_manual_gear  = false
                manual_gear     = 0
                gear_immediate  = true
            }

[<MessagePackObject(true)>]
type Pose = 
    {
       position :  float32[] 
       orientation : float32[]
    }

[<MessagePackObject(true)>]
type Twist = 
    {
       
       linear  : float32[] 
       angular : float32[]
    }

[<MessagePackObject(true)>]
type Accelerations = 
    {
       
       linear  : float32[] 
       angular : float32[]
    }

[<MessagePackObject(true)>]
type Kinematics_State(pose:Pose,twist:Twist,accelerations:Accelerations) = 
    member x.pose = pose
    member x.twist = twist
    member x.accelerations = accelerations

[<MessagePackObject(true)>]
type CarState =
    {
        speed : float32
        gear : int
        rpm : float32
        maxrpm : float32
        handbrake : bool
        Kinematics_State : Kinematics_State
        timestamp : uint64
    }

type AirSimCar(options) =
    inherit Client(options) 

    member this.getServerVersion() = 
        let name = nameof this.getServerVersion
        base.Call<_,int>(name,[||]) 

    member this.enableApiControl(isEnabled,?vehicle_name:string) =        
        let name = nameof this.enableApiControl
        let req:obj[] = [|isEnabled; defaultArg vehicle_name ""|]
        base.Call<_,unit>(name,req)

    member this.isApiControlEnabled(?vehicle_name:string) =        
        let name = nameof this.isApiControlEnabled
        let vehicle_name = defaultArg vehicle_name ""
        base.Call<_,bool>(name,[|vehicle_name|])

    member this.armDisarm(arm:bool,?vehicle_name:string) =        
        let name = nameof this.armDisarm
        let vehicle_name = defaultArg vehicle_name ""
        let req:obj[] = [|arm; vehicle_name|]
        base.Call<_,bool>(name,req)       
        
    member this.reset() =        
        let name = nameof this.reset
        base.Call<_,unit>(name,[||])        

    member this.setCarControls(car_controls:CarControls,?vehicle_name:string) =
        let name = nameof this.setCarControls  
        let vehicle_name = defaultArg vehicle_name ""
        let req:obj[] = [|car_controls; vehicle_name|]
        base.Call<_,unit>(name,req)        
        

let c1 = new AirSimCar(options)
try
    c1.Connect(address,port)
    //c1.getServerVersion() |> Async.AwaitTask |> Async.RunSynchronously
    //c1.enableApiControl(true) |> Async.AwaitTask |> Async.RunSynchronously
    //c1.isApiControlEnabled() |> Async.AwaitTask |> Async.RunSynchronously
    //c1.armDisarm(true) |> Async.AwaitTask |> Async.RunSynchronously
    c1.reset() |> Async.AwaitTask |> Async.RunSynchronously
    c1.setCarControls({CarControls.Default with throttle = 1.f }) |> Async.AwaitTask |> Async.RunSynchronously
finally
    (c1 :> IDisposable).Dispose()



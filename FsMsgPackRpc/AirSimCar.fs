namespace AirSimCar
open System
open MessagePack
open MessagePack.FSharp
open MessagePack.Resolvers
open FsMsgPackRpc

[<MessagePackObject(true)>]
type Vector3r =
    {
        x_val : float
        y_val : float
        z_val : float
    }
    with 
        member x.ToArray() = [|x.x_val; x.y_val; x.z_val|]

[<MessagePackObject(true)>]
type Quaternionr =
    {
        w_val : float
        x_val : float
        y_val : float
        z_val : float
    }
    with 
        member x.ToArray() = [|x.x_val; x.y_val; x.z_val; x.z_val|]

[<MessagePackObject(true)>]
type CarControls =
    {
        throttle        : float
        steering        : float
        brake           : float
        handbrake       : bool
        is_manual_gear  : bool
        manual_gear     : int
        gear_immediate  : bool
    }
    with
        static member Default =
            {
                throttle        = 0.
                steering        = 0.
                brake           = 0.
                handbrake       = false
                is_manual_gear  = false
                manual_gear     = 0
                gear_immediate  = true
            }

[<MessagePackObject(true)>]
type Pose = 
    {
       position :  Vector3r 
       orientation : Quaternionr
    }

[<MessagePackObject(true)>]
type Twist = 
    {       
       linear  : Vector3r
       angular : Vector3r
    }

[<MessagePackObject(true)>]
type Accelerations = 
    {
       linear  : Vector3r
       angular : Vector3r
    }

[<MessagePackObject(true)>]
type KinematicsState =
    {
        position : Vector3r
        orientation : Quaternionr
        linear_velocity : Vector3r
        angular_velocity : Vector3r
        linear_acceleration : Vector3r
        angular_acceleration : Vector3r
    }

[<MessagePackObject(true)>]
type CollisionInfo =
    {
        has_collided : bool
        normal : Vector3r
        impact_point : Vector3r
        position : Vector3r
        penetration_depth : float
        time_stamp : float
        object_name : string
        object_id : int
    }

[<MessagePackObject(true)>]
type CarState =
    {
        speed : float
        gear : int
        rpm : float
        maxrpm : float
        handbrake : bool
        kinematics_estimated : KinematicsState
        timestamp : uint64
    }

[<MessagePackObject>]
type ImageType = 
    | Scene = 0
    | DepthPlanar = 1
    | DepthPerspective = 2
    | DepthVis = 3
    | DisparityNormalized = 4
    | Segmentation = 5
    | SurfaceNormals = 6
    | Infrared = 7
    | OpticalFlow = 8
    | OpticalFlowVis = 9

[<MessagePackObject(true)>]
type ImageRequest = 
    {
        camera_name : string
        image_type : ImageType
        pixels_as_float : bool
        compress : bool
    }

[<MessagePackObject(true)>]
type ImageResponse =
    {
        image_data_uint8 : uint8[]
        image_data_float : float32[]
        camera_name : string
        camera_position : Vector3r
        camera_orientation : Quaternionr
        time_stamp : uint64
        message : string
        pixels_as_float : bool
        compress : bool
        width : int
        height : int
        image_type : ImageType
    }

module Defaults =

    let address = "localhost"
    let port = 41451
    let resolver = Resolvers.CompositeResolver.Create(
                    FSharpResolver.Instance,
                    StandardResolver.Instance)
    let options = MessagePackSerializerOptions.Standard.WithResolver(resolver)

type CarClient(options) =
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

    member this.simGetImage(camera_name:string, image_type:ImageType,?vehicle_name:string,?external:bool) =
        let name = nameof this.simGetImage
        let vehicle_name = defaultArg vehicle_name ""
        let external = defaultArg external false
        let req:obj[] = [|camera_name; image_type; vehicle_name; external|]
        base.Call<_,uint8[]>(name,req)    
        
    member this.simGetImages(requests:ImageRequest[],?vehicle_name:string,?external:bool) =
        let name = nameof this.simGetImages  
        let vehicle_name = defaultArg vehicle_name ""
        let external = defaultArg external false
        let req:obj[] = [|requests; vehicle_name; external|]
        base.Call<_,ImageResponse[]>(name,req)  
        
    member this.getCarState(?vehicle_name:string) =
        let name = nameof this.getCarState  
        let vehicle_name = defaultArg vehicle_name ""
        let req:obj[] = [|vehicle_name|]
        base.Call<_,CarState>(name,req) 
        
    member this.simGetCollisionInfo(?vehicle_name:string) =
        let name = nameof this.simGetCollisionInfo
        let vehicle_name = defaultArg vehicle_name ""
        let req:obj[] = [|vehicle_name|]
        base.Call<_,CollisionInfo>(name,req)  

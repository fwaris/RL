namespace AirSimCar
//MessagePack-rpc client for the the AirSim Car environment
//AirSim is a simulation environmnet built on top of the Unreal game engine
//see: https://microsoft.github.io/AirSim/
open System
open MessagePack
open MessagePack.FSharp
open MessagePack.Resolvers
open FsMsgPackRpc
open System.Numerics

[<MessagePackObject(true)>]
type Vector3r =
    {
        x_val : float
        y_val : float
        z_val : float
    }
    with 
        member x.ToArray() = [|x.x_val; x.y_val; x.z_val|]
        member x.ToVector3() = Vector3(float32 x.x_val,float32 x.y_val, float32 x.z_val)
        static member Default = {x_val=0.; y_val=0.; z_val=0}
        static member FromVector3 (v:Vector3) = {x_val=float v.X; y_val=float v.Y; z_val=float v.Z}

[<MessagePackObject(true)>]
type Quaternionr =
    {
        w_val : float
        x_val : float
        y_val : float
        z_val : float
    }
    with 
        member x.ToQuaternion() = Quaternion(float32 x.x_val, float32 x.y_val, float32 x.z_val, float32 x.w_val)
        static member Default = {w_val= 1.; x_val=0.; y_val=0.; z_val=0.}
        static member FromQuaternion(q:Quaternion) = {w_val=float q.W; x_val=float q.X; y_val=float q.Y; z_val=float q.Z}

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
    with 
        static member Default = 
            {
                position = Vector3r.Default
                orientation = Quaternionr.Default
            }

[<MessagePackObject(true)>]
type CameraInfo = {
    fov : float
    pose : Pose
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
    static member Default = 
        {
            position = Vector3r.Default
            orientation = Quaternionr.Default
            linear_velocity = Vector3r.Default
            angular_velocity = Vector3r.Default
            linear_acceleration = Vector3r.Default
            angular_acceleration = Vector3r.Default
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
    with 
        static member Default = 
                            {
                                has_collided  = false
                                normal = Vector3r.Default
                                impact_point = Vector3r.Default
                                position =  Vector3r.Default
                                penetration_depth = 0.0
                                time_stamp = 0.0
                                object_name = ""
                                object_id = 0
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

//[<MessagePackObject>]
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

///client for the AirSim 'car' environment - only the methods required for the DDQN sample have been implemented
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
        
    member this.simListSceneObjects(?name_regex:string) =        
        let name = nameof this.simListSceneObjects
        let name_regex = defaultArg name_regex ".*"
        let req:obj[] = [|name_regex|]
        base.Call<_,string list>(name,req)  

    member this.simListAssets() =        
        let name = nameof this.simListAssets
        let req:obj[] = [||]
        base.Call<_,string list>(name,req)  

    member this.simGetCameraInfo(camera_name:string,?vehicle_name:string) =
        let name = nameof this.simGetCameraInfo  
        let vehicle_name = defaultArg vehicle_name ""
        let req:obj[] = [|camera_name; vehicle_name|]
        base.Call<_,CameraInfo>(name,req)    

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

     member this.simGetGroundTruthKinematics(?vehicle_name:string) =
        let name = nameof this.simGetGroundTruthKinematics
        let vehicle_name = defaultArg vehicle_name ""
        let req:obj[] = [|vehicle_name|]
        base.Call<_,KinematicsState>(name,req)    
        
    member this.simSetObjectPose(object_name:string, pose:Pose, teleport:bool) =
        let name = nameof this.simSetObjectPose     
        let req:obj[] = [|object_name; pose; teleport|]
        base.Call<_,bool>(name,req)  

    member this.simSetKinematics(state:KinematicsState, ignore_collision:bool, ?vehicle_name:string) =
        let name = nameof this.simSetKinematics     
        let vehicle_name = defaultArg vehicle_name ""
        let req:obj[] = [|state;ignore_collision; vehicle_name|]
        base.Call<_,unit>(name,req)  

    member this.simSetCameraPose(camera_name:string, pose:Pose, ?vehicle_name: string, ?external:bool) =
        let name = nameof this.simSetCameraPose     
        let vehicle_name = defaultArg vehicle_name ""
        let external = defaultArg external false
        let req:obj[] = [|camera_name; pose; vehicle_name; external|]
        base.Call<_,unit>(name,req)  

    member this.simGetObjectPose(object_name:string) =
        let name = nameof this.simGetObjectPose     
        let req:obj[] = [|object_name|]
        base.Call<_,Pose>(name,req)  

    member this.simGetCollisionInfo(?vehicle_name:string) =
        let name = nameof this.simGetCollisionInfo
        let vehicle_name = defaultArg vehicle_name ""
        let req:obj[] = [|vehicle_name|]
        base.Call<_,CollisionInfo>(name,req)  

    member this.simIsPause() =
        let name = nameof this.simIsPause
        let req:obj[] = [||]
        base.Call<_,bool>(name,req)  

    member this.simPause(is_paused:bool) =
        let name = nameof this.simPause
        let req:obj[] = [|is_paused|]
        base.Call<_,unit>(name,req)  

    member this.simSpawnObject(
        object_name:string,
        asset_name:string,
        pose:Pose,
        scale:Vector3r,
        ?physics_enabled:bool,
        ?is_blueprint:bool
    ) =
        let name = nameof this.simSpawnObject
        let physics_enabled = defaultArg physics_enabled false
        let is_blueprint = defaultArg is_blueprint false
        let req:obj[] =
            [|
                object_name
                asset_name
                pose
                scale
                physics_enabled
                is_blueprint
            |]
        base.Call<_,string>(name,req)  

    member this.simDestroyObject(object_name:string) =
        let name = nameof this.simDestroyObject
        let req:obj[] = [|object_name|]
        base.Call<_,bool>(name,req)  

    member this.simPlotLineList(
        points:Vector3r list,
        ?color_rgba:(float*float*float*float),
        ?thickness:float,
        ?duration:float,
        ?is_persistent:bool
        ) =
        let name = nameof this.simPlotLineList
        let (r,g,b,a) = defaultArg color_rgba (1.,0.,0.,1.)
        let thickness = defaultArg thickness 5.0
        let duration = defaultArg duration -1.0
        let is_persistent = defaultArg is_persistent false
        let req:obj[] = 
            [|
                List.toArray points
                [|r;g;b;a|]
                thickness
                duration
                is_persistent
            |]
        base.Call<_,unit>(name,req)  

    member this.simPlotPoints(
        points:Vector3r list,
        ?color_rgba:(float*float*float*float),
        ?size:float,
        ?duration:float,
        ?is_persistent:bool
        ) =
        let name = nameof this.simPlotPoints
        let (r,g,b,a) = defaultArg color_rgba (1.,0.,0.,1.)
        let size = defaultArg size 5.0
        let duration = defaultArg duration -1.0
        let is_persistent = defaultArg is_persistent false
        let req:obj[] = 
            [|
                List.toArray points
                [|r;g;b;a|]
                size
                duration
                is_persistent
            |]
        base.Call<_,unit>(name,req)  

    member this.simPlotStrings(
        strs:string list, 
        positions: Vector3r list, 
        ?scale:float,
        ?color_rgba:(float*float*float*float),
        ?duration:float
        ) =
        let name = nameof this.simPlotStrings
        let scale = defaultArg scale 5.0
        let (r,g,b,a) = defaultArg color_rgba (1.,0.,0.,1.)
        let duration = defaultArg duration -1.0
        let req:obj[] = 
            [|
                List.toArray strs
                List.toArray positions
                scale
                [|r;g;b;a|]                
                duration
            |]
        base.Call<_,unit>(name,req)  



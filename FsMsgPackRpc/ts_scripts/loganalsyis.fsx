let s = """
Welcome to F# Interactive for .NET Core in Visual Studio. To execute code, either
  1. Use 'Send to Interactive' (Alt-Enter or right-click) from an F# script. The F# Interactive process will
     use any global.json settings associated with that script.
  2. Press 'Enter' to start. The F# Interactive process will use default settings.
> 

Microsoft (R) F# Interactive version 12.0.4.0 for F# 6.0
Copyright (c) Microsoft Corporation. All Rights Reserved.

For help type #help;;

> [Loading C:\Users\Faisa\.packagemanagement\nuget\Projects\19868--edcc242e-557a-4a1d-bdbc-88a3ad179cb4\Project.fsproj.fsx
 Loading C:\Users\Faisa\source\repos\FsMsgPackRpc\FsMsgPackRpc\scripts\../FsMsgPackRpc.fs
 Loading C:\Users\Faisa\source\repos\FsMsgPackRpc\FsMsgPackRpc\scripts\../AirSimCar.fs
 Loading C:\Users\Faisa\source\repos\FsMsgPackRpc\FsMsgPackRpc\scripts\../CarEnvironment.fs
 Loading C:\Users\Faisa\source\repos\FsMsgPackRpc\FsMsgPackRpc\scripts\../DDQN.fs
 Loading C:\Users\Faisa\source\repos\FsMsgPackRpc\FsMsgPackRpc\scripts\packages.fsx]
Binding session to 'C:/Users/Faisa/.nuget/packages/torchsharp/0.97.3/lib/netcoreapp3.1/TorchSharp.dll'...
module FSI_0003.Project.fsproj

namespace FSI_0003.FsMsgPackRpc
  type ServerResp =
    | Data of obj
    | Error of obj
  type Msg =
    | Req of (int * (System.Type * AsyncReplyChannel<ServerResp>))
    | Resp of int * obj * byte[]
  val createAgent:
    options: MessagePack.MessagePackSerializerOptions ->
      cts: System.Threading.CancellationTokenSource ->
      inbox: MailboxProcessor<Msg> -> Async<unit>
  type Client =
    interface System.IDisposable
    new: options: MessagePack.MessagePackSerializerOptions -> Client
    member
      Call: method: string * req: 'req -> System.Threading.Tasks.Task<'resp>
    member Connect: address: string * port: int -> unit
    member Disconnect: unit -> unit
    member
      Notify: method: string * req: 'req -> System.Threading.Tasks.Task<unit>
    member TcpClient: System.Net.Sockets.TcpClient

namespace FSI_0003.AirSimCar
  type Vector3r =
    {
      x_val: float
      y_val: float
      z_val: float
    }
    member ToArray: unit -> float[]
    static member Default: Vector3r
  type Quaternionr =
    {
      w_val: float
      x_val: float
      y_val: float
      z_val: float
    }
    member ToArray: unit -> float[]
    static member Default: Quaternionr
  type CarControls =
    {
      throttle: float
      steering: float
      brake: float
      handbrake: bool
      is_manual_gear: bool
      manual_gear: int
      gear_immediate: bool
    }
    static member Default: CarControls
  type Pose =
    {
      position: Vector3r
      orientation: Quaternionr
    }
  type Twist =
    {
      linear: Vector3r
      angular: Vector3r
    }
  type Accelerations =
    {
      linear: Vector3r
      angular: Vector3r
    }
  type KinematicsState =
    {
      position: Vector3r
      orientation: Quaternionr
      linear_velocity: Vector3r
      angular_velocity: Vector3r
      linear_acceleration: Vector3r
      angular_acceleration: Vector3r
    }
    static member Default: KinematicsState
  type CollisionInfo =
    {
      has_collided: bool
      normal: Vector3r
      impact_point: Vector3r
      position: Vector3r
      penetration_depth: float
      time_stamp: float
      object_name: string
      object_id: int
    }
  type CarState =
    {
      speed: float
      gear: int
      rpm: float
      maxrpm: float
      handbrake: bool
      kinematics_estimated: KinematicsState
      timestamp: uint64
    }
  [<Struct>]
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
  type ImageRequest =
    {
      camera_name: string
      image_type: ImageType
      pixels_as_float: bool
      compress: bool
    }
  type ImageResponse =
    {
      image_data_uint8: uint8[]
      image_data_float: float32[]
      camera_name: string
      camera_position: Vector3r
      camera_orientation: Quaternionr
      time_stamp: uint64
      message: string
      pixels_as_float: bool
      compress: bool
      width: int
      height: int
      image_type: ImageType
    }
  val address: string
  val port: int
  val resolver: MessagePack.IFormatterResolver
  val options: MessagePack.MessagePackSerializerOptions
  type CarClient =
    inherit FsMsgPackRpc.Client
    new: options: MessagePack.MessagePackSerializerOptions -> CarClient
    member
      armDisarm: arm: bool * ?vehicle_name: string ->
                   System.Threading.Tasks.Task<bool>
    member
      enableApiControl: isEnabled: 'a * ?vehicle_name: string ->
                          System.Threading.Tasks.Task<unit>
    member
      getCarState: ?vehicle_name: string ->
                     System.Threading.Tasks.Task<CarState>
    member getServerVersion: unit -> System.Threading.Tasks.Task<int>
    member
      isApiControlEnabled: ?vehicle_name: string ->
                             System.Threading.Tasks.Task<bool>
    member reset: unit -> System.Threading.Tasks.Task<unit>
    member
      setCarControls: car_controls: CarControls * ?vehicle_name: string ->
                        System.Threading.Tasks.Task<unit>
    member
      simGetCollisionInfo: ?vehicle_name: string ->
                             System.Threading.Tasks.Task<CollisionInfo>
    member
      simGetGroundTruthKinematics: ?vehicle_name: string ->
                                     System.Threading.Tasks.Task<KinematicsState>
    member
      simGetImage: camera_name: string * image_type: ImageType *
                   ?vehicle_name: string * ?external: bool ->
                     System.Threading.Tasks.Task<uint8[]>
    member
      simGetImages: requests: ImageRequest[] * ?vehicle_name: string *
                    ?external: bool ->
                      System.Threading.Tasks.Task<ImageResponse[]>
    member
      simGetObjectPose: object_name: string ->
                          System.Threading.Tasks.Task<Pose>
    member
      simListSceneObjects: ?name_regex: string ->
                             System.Threading.Tasks.Task<string list>
    member
      simSetKinematics: state: KinematicsState * ?ignore_collision: bool *
                        ?vehicle_name: string ->
                          System.Threading.Tasks.Task<unit>
    member
      simSetObjectPose: object_name: string * pose: Pose * teleport: bool ->
                          System.Threading.Tasks.Task<bool>

module FSI_0003.CarEnvironment
val discreteActions: int
type RLState =
  {
    Pose: TorchSharp.torch.Tensor
    Speed: float
    Collision: bool
    DepthImage: TorchSharp.torch.Tensor
    PrevDepthImage: TorchSharp.torch.Tensor
    WasReset: bool
  }
  static member Default: RLState
type DoneReason =
  | LowReward
  | Collision
  | Stuck
  | NotDone
val doAction:
  c: AirSimCar.CarClient ->
    action: int ->
    carCtrl: AirSimCar.CarControls ->
    waitMs: int -> Async<AirSimCar.CarControls>
val imageRequest: AirSimCar.ImageRequest[]
val transformImage: resp: AirSimCar.ImageResponse -> TorchSharp.torch.Tensor
val getObservations:
  c: AirSimCar.CarClient ->
    prevState: RLState -> System.Threading.Tasks.Task<RLState>
val roadPts: TorchSharp.torch.Tensor list
val computeReward:
  state: RLState ->
    ctrls: AirSimCar.CarControls -> float * DoneReason * RLState
val step:
  c: AirSimCar.CarClient ->
    state: RLState * ctrls: AirSimCar.CarControls ->
      action: int ->
      waitMs: int ->
      System.Threading.Tasks.Task<RLState * AirSimCar.CarControls * float *
                                  DoneReason>
val rng: System.Random
val randRoadPoint:
  unit ->
    TorchSharp.torch.Tensor * TorchSharp.torch.Tensor *
    TorchSharp.torch.Tensor
val randPose: unit -> AirSimCar.Pose
val carId: string
val initCar: c: AirSimCar.CarClient -> System.Threading.Tasks.Task<unit>
val startRandomAgent: go: bool ref -> Async<unit>

module FSI_0003.DDQN
type DDQNModel =
  {
    Target: TorchSharp.Fun.IModel
    Online: TorchSharp.Fun.IModel
  }
type Experience =
  {
    State: TorchSharp.torch.Tensor
    NextState: TorchSharp.torch.Tensor
    Action: int
    Reward: float32
    Done: bool
  }
type ExperienceBuffer =
  {
    Buffer: FSharpx.Collections.RandomAccessList<Experience>
    Max: int
  }
type Exploration =
  {
    Decay: float
    Min: float
  }
  static member Default: Exploration
type Step =
  {
    Num: int
    ExplorationRate: float
  }
type DDQN =
  {
    Model: DDQNModel
    Gamma: float32
    Exploration: Exploration
    Actions: int
    Device: TorchSharp.torch.Device
  }
module DDQNModel =
  val create: fmodel: (unit -> TorchSharp.Fun.IModel) -> DDQNModel
  val sync: models: DDQNModel -> device: TorchSharp.torch.Device -> unit
  val save: file: string -> ddqn: DDQNModel -> unit
  val load:
    fmodel: (unit -> TorchSharp.Fun.IModel) -> file: string -> DDQNModel
module Experience =
  val createBuffer: maxExperiance: int -> ExperienceBuffer
  val append: exp: Experience -> buff: ExperienceBuffer -> ExperienceBuffer
  val sample: n: int -> buff: ExperienceBuffer -> Experience[]
  val recall:
    n: int ->
      buff: ExperienceBuffer ->
      TorchSharp.torch.Tensor * TorchSharp.torch.Tensor * float32[] * int[] *
      bool[]
  type Tser =
    int * int64[] * List<float32[] * float32[] * int * float32 * bool>
  val save: path: string -> buff: ExperienceBuffer -> unit
  val load: path: string -> ExperienceBuffer
module DDQN =
  val rand: unit -> float
  val randint: n: int -> int
  val updateStep: exp: Exploration -> step: Step -> Step
  val create:
    model: DDQNModel ->
      gamma: float32 ->
      exploration: Exploration ->
      actions: int -> device: TorchSharp.torch.Device -> DDQN
  val selectAction:
    state: TorchSharp.torch.Tensor -> ddqn: DDQN -> step: Step -> int
  val actionIdx:
    actions: TorchSharp.torch.Tensor -> TorchSharp.torch.TensorIndex[]
  val td_estimate:
    state: TorchSharp.torch.Tensor ->
      actions: int[] -> ddqn: DDQN -> TorchSharp.torch.Tensor
  val td_target:
    reward: float32[] ->
      next_state: TorchSharp.torch.Tensor ->
      isDone: bool[] -> ddqn: DDQN -> TorchSharp.torch.Tensor

module FSI_0003.Packages
val userProfile: string
val nugetPath: string
val openCvLibPath: string
val path: string
val path': string

[Loading C:\Users\Faisa\.packagemanagement\nuget\Projects\19868--edcc242e-557a-4a1d-bdbc-88a3ad179cb4\Project.fsproj.fsx]
module FSI_0004.Project.fsproj

[Loading C:\Users\Faisa\source\repos\FsMsgPackRpc\FsMsgPackRpc\TsData.fs]
module FSI_0005.TsData
type Bar =
  {
    Open: float
    High: float
    Low: float
    Close: float
    Volume: float
    Time: System.DateTime
  }
val loadBars: file: string -> Bar[]

[Loading C:\Users\Faisa\source\repos\FsMsgPackRpc\FsMsgPackRpc\RL.fs]
module FSI_0006.RL
type Agent<'parms,'env,'state> =
  {
    doAction: ('parms -> 'env -> 'state -> int -> 'state)
    getObservations: ('parms -> 'env -> 'state -> 'state)
    computeRewards: ('parms -> 'env -> 'state -> int -> 'state * bool * float)
  }
type Policy<'parms,'state> =
  {
    selectAction: ('parms -> 'state -> Policy<'parms,'state> * int)
    update:
      ('parms -> 'state -> bool -> float -> Policy<'parms,'state> * 'state)
    sync: ('parms -> 'state -> unit)
  }
val step:
  parms: 'a ->
    env: 'b ->
    agent: Agent<'a,'b,'c> ->
    policy: Policy<'a,'c> * s0: 'c -> Policy<'a,'c> * 'c

Binding session to 'C:/Users/Faisa/.nuget/packages/torchsharp.fun/0.60.0/lib/net6.0/TorchSharp.Fun.dll'...
Binding session to 'C:/Users/Faisa/.nuget/packages/fspickler/5.3.2/lib/netstandard2.0/FsPickler.dll'...
val device: TorchSharp.torch.Device = cuda
val root: string = "E:\s\tradestation"
val (@@) : a: string -> b: string -> string
val fn: string = "E:\s\tradestation\mes_5_min.bin"
val fnTest: string = "E:\s\tradestation\mes_5_min_test.bin"
val data: TsData.Bar[] =
  [|{ Open = 3989.25
      High = 3990.25
      Low = 3985.25
      Close = 3985.25
      Volume = 1316.0
      Time = 7/21/2022 6:05:00 PM }; { Open = 3985.0
                                       High = 3985.75
                                       Low = 3982.75
                                       Close = 3984.25
                                       Volume = 979.0
                                       Time = 7/21/2022 6:10:00 PM };
    { Open = 3984.5
      High = 3985.0
      Low = 3982.75
      Close = 3984.5
      Volume = 637.0
      Time = 7/21/2022 6:15:00 PM }; { Open = 3984.25
                                       High = 3987.25
                                       Low = 3983.25
                                       Close = 3986.75
                                       Volume = 689.0
                                       Time = 7/21/2022 6:20:00 PM };
    { Open = 3987.0
      High = 3989.5
      Low = 3986.25
      Close = 3986.5
      Volume = 1259.0
      Time = 7/21/2022 6:25:00 PM }; { Open = 3986.75
                                       High = 3989.25
                                       Low = 3986.25
                                       Close = 3988.25
                                       Volume = 523.0
                                       Time = 7/21/2022 6:30:00 PM };
    { Open = 3988.0
      High = 3990.0
      Low = 3987.25
      Close = 3988.5
      Volume = 782.0
      Time = 7/21/2022 6:35:00 PM }; { Open = 3988.5
                                       High = 3988.5
                                       Low = 3986.0
                                       Close = 3986.75
                                       Volume = 797.0
                                       Time = 7/21/2022 6:40:00 PM };
    { Open = 3986.75
      High = 3987.25
      Low = 3986.0
      Close = 3986.75
      Volume = 311.0
      Time = 7/21/2022 6:45:00 PM }; { Open = 3986.75
                                       High = 3987.75
                                       Low = 3985.5
                                       Close = 3987.25
                                       Volume = 282.0
                                       Time = 7/21/2022 6:50:00 PM };
    { Open = 3987.25
      High = 3987.5
      Low = 3985.5
      Close = 3986.25
      Volume = 200.0
      Time = 7/21/2022 6:55:00 PM }; { Open = 3986.5
                                       High = 3987.25
                                       Low = 3986.0
                                       Close = 3986.5
                                       Volume = 237.0
                                       Time = 7/21/2022 7:00:00 PM };
    { Open = 3986.25
      High = 3986.75
      Low = 3985.0
      Close = 3986.25
      Volume = 346.0
      Time = 7/21/2022 7:05:00 PM }; { Open = 3986.0
                                       High = 3986.5
                                       Low = 3985.25
                                       Close = 3985.75
                                       Volume = 172.0
                                       Time = 7/21/2022 7:10:00 PM };
    { Open = 3985.75
      High = 3986.75
      Low = 3985.5
      Close = 3985.75
      Volume = 174.0
      Time = 7/21/2022 7:15:00 PM }; { Open = 3985.5
                                       High = 3986.25
                                       Low = 3984.25
                                       Close = 3985.0
                                       Volume = 563.0
                                       Time = 7/21/2022 7:20:00 PM };
    { Open = 3985.0
      High = 3985.5
      Low = 3984.0
      Close = 3985.25
      Volume = 197.0
      Time = 7/21/2022 7:25:00 PM }; { Open = 3985.25
                                       High = 3987.25
                                       Low = 3985.0
                                       Close = 3987.0
                                       Volume = 384.0
                                       Time = 7/21/2022 7:30:00 PM };
    { Open = 3986.75
      High = 3988.25
      Low = 3985.75
      Close = 3987.0
      Volume = 795.0
      Time = 7/21/2022 7:35:00 PM }; { Open = 3987.25
                                       High = 3987.25
                                       Low = 3985.0
                                       Close = 3985.5
                                       Volume = 384.0
                                       Time = 7/21/2022 7:40:00 PM };
    { Open = 3985.5
      High = 3987.0
      Low = 3985.5
      Close = 3986.5
      Volume = 184.0
      Time = 7/21/2022 7:45:00 PM }; { Open = 3986.5
                                       High = 3987.25
                                       Low = 3985.25
                                       Close = 3986.25
                                       Volume = 454.0
                                       Time = 7/21/2022 7:50:00 PM };
    { Open = 3986.25
      High = 3986.25
      Low = 3983.5
      Close = 3984.25
      Volume = 578.0
      Time = 7/21/2022 7:55:00 PM }; { Open = 3984.0
                                       High = 3985.0
                                       Low = 3983.5
                                       Close = 3984.25
                                       Volume = 488.0
                                       Time = 7/21/2022 8:00:00 PM };
    { Open = 3984.25
      High = 3984.25
      Low = 3980.5
      Close = 3983.0
      Volume = 1650.0
      Time = 7/21/2022 8:05:00 PM }; { Open = 3983.0
                                       High = 3983.5
                                       Low = 3981.75
                                       Close = 3981.75
                                       Volume = 668.0
                                       Time = 7/21/2022 8:10:00 PM };
    { Open = 3981.5
      High = 3982.5
      Low = 3980.5
      Close = 3982.25
      Volume = 618.0
      Time = 7/21/2022 8:15:00 PM }; { Open = 3982.0
                                       High = 3985.0
                                       Low = 3981.75
                                       Close = 3984.0
                                       Volume = 938.0
                                       Time = 7/21/2022 8:20:00 PM };
    { Open = 3984.25
      High = 3985.5
      Low = 3983.75
      Close = 3984.75
      Volume = 369.0
      Time = 7/21/2022 8:25:00 PM }; { Open = 3985.0
                                       High = 3986.0
                                       Low = 3983.75
                                       Close = 3983.75
                                       Volume = 631.0
                                       Time = 7/21/2022 8:30:00 PM };
    { Open = 3983.75
      High = 3986.0
      Low = 3983.75
      Close = 3984.75
      Volume = 385.0
      Time = 7/21/2022 8:35:00 PM }; { Open = 3984.75
                                       High = 3986.5
                                       Low = 3983.0
                                       Close = 3986.25
                                       Volume = 598.0
                                       Time = 7/21/2022 8:40:00 PM };
    { Open = 3986.25
      High = 3986.5
      Low = 3984.0
      Close = 3984.25
      Volume = 337.0
      Time = 7/21/2022 8:45:00 PM }; { Open = 3984.5
                                       High = 3985.75
                                       Low = 3984.0
                                       Close = 3985.25
                                       Volume = 325.0
                                       Time = 7/21/2022 8:50:00 PM };
    { Open = 3985.25
      High = 3986.75
      Low = 3985.25
      Close = 3986.25
      Volume = 397.0
      Time = 7/21/2022 8:55:00 PM }; { Open = 3986.0
                                       High = 3987.5
                                       Low = 3985.75
                                       Close = 3987.25
                                       Volume = 555.0
                                       Time = 7/21/2022 9:00:00 PM };
    { Open = 3987.25
      High = 3987.25
      Low = 3983.25
      Close = 3984.0
      Volume = 984.0
      Time = 7/21/2022 9:05:00 PM }; { Open = 3984.0
                                       High = 3985.0
                                       Low = 3983.5
                                       Close = 3984.0
                                       Volume = 525.0
                                       Time = 7/21/2022 9:10:00 PM };
    { Open = 3983.75
      High = 3984.0
      Low = 3982.5
      Close = 3983.75
      Volume = 464.0
      Time = 7/21/2022 9:15:00 PM }; { Open = 3983.5
                                       High = 3984.0
                                       Low = 3981.25
                                       Close = 3982.75
                                       Volume = 687.0
                                       Time = 7/21/2022 9:20:00 PM };
    { Open = 3982.5
      High = 3984.25
      Low = 3982.25
      Close = 3983.25
      Volume = 475.0
      Time = 7/21/2022 9:25:00 PM }; { Open = 3983.25
                                       High = 3985.75
                                       Low = 3983.25
                                       Close = 3984.75
                                       Volume = 570.0
                                       Time = 7/21/2022 9:30:00 PM };
    { Open = 3984.75
      High = 3987.0
      Low = 3984.75
      Close = 3986.75
      Volume = 744.0
      Time = 7/21/2022 9:35:00 PM }; { Open = 3987.0
                                       High = 3989.25
                                       Low = 3986.0
                                       Close = 3988.5
                                       Volume = 1144.0
                                       Time = 7/21/2022 9:40:00 PM };
    { Open = 3988.5
      High = 3991.5
      Low = 3988.25
      Close = 3990.0
      Volume = 1503.0
      Time = 7/21/2022 9:45:00 PM }; { Open = 3989.75
                                       High = 3990.0
                                       Low = 3987.0
                                       Close = 3987.0
                                       Volume = 965.0
                                       Time = 7/21/2022 9:50:00 PM };
    { Open = 3987.25
      High = 3988.5
      Low = 3986.0
      Close = 3987.5
      Volume = 607.0
      Time = 7/21/2022 9:55:00 PM }; { Open = 3987.25
                                       High = 3988.5
                                       Low = 3986.75
                                       Close = 3988.25
                                       Volume = 475.0
                                       Time = 7/21/2022 10:00:00 PM };
    { Open = 3988.25
      High = 3989.25
      Low = 3987.25
      Close = 3988.75
      Volume = 583.0
      Time = 7/21/2022 10:05:00 PM }; { Open = 3988.5
                                        High = 3989.0
                                        Low = 3986.75
                                        Close = 3987.0
                                        Volume = 322.0
                                        Time = 7/21/2022 10:10:00 PM };
    { Open = 3987.0
      High = 3987.0
      Low = 3985.25
      Close = 3985.25
      Volume = 589.0
      Time = 7/21/2022 10:15:00 PM }; { Open = 3985.25
                                        High = 3986.25
                                        Low = 3984.0
                                        Close = 3985.75
                                        Volume = 719.0
                                        Time = 7/21/2022 10:20:00 PM };
    { Open = 3986.0
      High = 3986.5
      Low = 3984.75
      Close = 3984.75
      Volume = 435.0
      Time = 7/21/2022 10:25:00 PM }; { Open = 3984.75
                                        High = 3985.5
                                        Low = 3983.5
                                        Close = 3984.25
                                        Volume = 555.0
                                        Time = 7/21/2022 10:30:00 PM };
    { Open = 3984.0
      High = 3985.0
      Low = 3983.75
      Close = 3983.75
      Volume = 320.0
      Time = 7/21/2022 10:35:00 PM }; { Open = 3983.75
                                        High = 3983.75
                                        Low = 3982.0
                                        Close = 3982.25
                                        Volume = 652.0
                                        Time = 7/21/2022 10:40:00 PM };
    { Open = 3982.5
      High = 3984.5
      Low = 3982.25
      Close = 3984.25
      Volume = 494.0
      Time = 7/21/2022 10:45:00 PM }; { Open = 3984.5
                                        High = 3984.75
                                        Low = 3983.25
                                        Close = 3983.75
                                        Volume = 422.0
                                        Time = 7/21/2022 10:50:00 PM };
    { Open = 3983.5
      High = 3985.5
      Low = 3983.0
      Close = 3984.25
      Volume = 777.0
      Time = 7/21/2022 10:55:00 PM }; { Open = 3984.25
                                        High = 3984.5
                                        Low = 3983.0
                                        Close = 3983.5
                                        Volume = 341.0
                                        Time = 7/21/2022 11:00:00 PM };
    { Open = 3983.5
      High = 3985.25
      Low = 3983.25
      Close = 3984.5
      Volume = 358.0
      Time = 7/21/2022 11:05:00 PM }; { Open = 3984.25
                                        High = 3984.75
                                        Low = 3983.5
                                        Close = 3984.5
                                        Volume = 263.0
                                        Time = 7/21/2022 11:10:00 PM };
    { Open = 3984.5
      High = 3985.5
      Low = 3984.25
      Close = 3985.0
      Volume = 275.0
      Time = 7/21/2022 11:15:00 PM }; { Open = 3985.0
                                        High = 3986.0
                                        Low = 3983.75
                                        Close = 3983.75
                                        Volume = 368.0
                                        Time = 7/21/2022 11:20:00 PM };
    { Open = 3983.5
      High = 3984.25
      Low = 3982.25
      Close = 3983.0
      Volume = 513.0
      Time = 7/21/2022 11:25:00 PM }; { Open = 3983.0
                                        High = 3985.0
                                        Low = 3982.75
                                        Close = 3984.75
                                        Volume = 335.0
                                        Time = 7/21/2022 11:30:00 PM };
    { Open = 3984.75
      High = 3986.0
      Low = 3984.75
      Close = 3985.75
      Volume = 423.0
      Time = 7/21/2022 11:35:00 PM }; { Open = 3985.5
                                        High = 3986.0
                                        Low = 3984.5
                                        Close = 3985.25
                                        Volume = 302.0
                                        Time = 7/21/2022 11:40:00 PM };
    { Open = 3985.0
      High = 3986.0
      Low = 3984.75
      Close = 3986.0
      Volume = 255.0
      Time = 7/21/2022 11:45:00 PM }; { Open = 3986.0
                                        High = 3986.75
                                        Low = 3985.5
                                        Close = 3986.0
                                        Volume = 402.0
                                        Time = 7/21/2022 11:50:00 PM };
    { Open = 3985.75
      High = 3986.0
      Low = 3984.75
      Close = 3985.0
      Volume = 279.0
      Time = 7/21/2022 11:55:00 PM }; { Open = 3985.25
                                        High = 3986.5
                                        Low = 3984.75
                                        Close = 3986.25
                                        Volume = 259.0
                                        Time = 7/22/2022 12:00:00 AM };
    { Open = 3986.5
      High = 3987.25
      Low = 3986.25
      Close = 3987.25
      Volume = 332.0
      Time = 7/22/2022 12:05:00 AM }; { Open = 3987.0
                                        High = 3987.5
                                        Low = 3986.5
                                        Close = 3987.25
                                        Volume = 190.0
                                        Time = 7/22/2022 12:10:00 AM };
    { Open = 3987.5
      High = 3988.0
      Low = 3986.5
      Close = 3986.5
      Volume = 393.0
      Time = 7/22/2022 12:15:00 AM }; { Open = 3986.5
                                        High = 3986.75
                                        Low = 3985.75
                                        Close = 3986.5
                                        Volume = 177.0
                                        Time = 7/22/2022 12:20:00 AM };
    { Open = 3986.75
      High = 3987.5
      Low = 3986.5
      Close = 3986.75
      Volume = 256.0
      Time = 7/22/2022 12:25:00 AM }; { Open = 3987.0
                                        High = 3987.0
                                        Low = 3985.75
                                        Close = 3986.0
                                        Volume = 171.0
                                        Time = 7/22/2022 12:30:00 AM };
    { Open = 3986.0
      High = 3986.5
      Low = 3984.75
      Close = 3984.75
      Volume = 332.0
      Time = 7/22/2022 12:35:00 AM }; { Open = 3985.0
                                        High = 3986.25
                                        Low = 3984.75
                                        Close = 3985.5
                                        Volume = 154.0
                                        Time = 7/22/2022 12:40:00 AM };
    { Open = 3985.5
      High = 3985.75
      Low = 3984.75
      Close = 3985.0
      Volume = 172.0
      Time = 7/22/2022 12:45:00 AM }; { Open = 3985.25
                                        High = 3985.75
                                        Low = 3984.75
                                        Close = 3985.0
                                        Volume = 227.0
                                        Time = 7/22/2022 12:50:00 AM };
    { Open = 3985.0
      High = 3985.25
      Low = 3984.5
      Close = 3984.75
      Volume = 168.0
      Time = 7/22/2022 12:55:00 AM }; { Open = 3985.0
                                        High = 3985.25
                                        Low = ...
                                        Close = ...
                                        Volume = ...
                                        Time = ... }; ...|]
val d2: TsData.Bar = { Open = 4369.0
                       High = 4369.5
                       Low = 4367.0
                       Close = 4367.0
                       Volume = 496.0
                       Time = 2/17/2022 5:00:00 PM }
val d1: TsData.Bar = { Open = 3989.25
                       High = 3990.25
                       Low = 3985.25
                       Close = 3985.25
                       Volume = 1316.0
                       Time = 7/21/2022 6:05:00 PM }
val dataTest: TsData.Bar[] =
  [|{ Open = 4117.5
      High = 4118.5
      Low = 4116.75
      Close = 4118.25
      Volume = 638.0
      Time = 7/31/2022 7:55:00 PM }; { Open = 4118.25
                                       High = 4119.25
                                       Low = 4116.75
                                       Close = 4117.0
                                       Volume = 560.0
                                       Time = 7/31/2022 8:00:00 PM };
    { Open = 4117.0
      High = 4117.75
      Low = 4114.5
      Close = 4115.25
      Volume = 1353.0
      Time = 7/31/2022 8:05:00 PM }; { Open = 4115.25
                                       High = 4116.75
                                       Low = 4114.5
                                       Close = 4115.0
                                       Volume = 809.0
                                       Time = 7/31/2022 8:10:00 PM };
    { Open = 4115.0
      High = 4116.0
      Low = 4114.25
      Close = 4115.25
      Volume = 600.0
      Time = 7/31/2022 8:15:00 PM }; { Open = 4115.25
                                       High = 4120.0
                                       Low = 4115.25
                                       Close = 4118.5
                                       Volume = 1648.0
                                       Time = 7/31/2022 8:20:00 PM };
    { Open = 4118.5
      High = 4119.25
      Low = 4117.5
      Close = 4118.5
      Volume = 436.0
      Time = 7/31/2022 8:25:00 PM }; { Open = 4118.25
                                       High = 4120.25
                                       Low = 4118.0
                                       Close = 4119.5
                                       Volume = 468.0
                                       Time = 7/31/2022 8:30:00 PM };
    { Open = 4119.75
      High = 4120.0
      Low = 4117.25
      Close = 4117.25
      Volume = 493.0
      Time = 7/31/2022 8:35:00 PM }; { Open = 4117.25
                                       High = 4118.5
                                       Low = 4116.25
                                       Close = 4116.75
                                       Volume = 582.0
                                       Time = 7/31/2022 8:40:00 PM };
    { Open = 4116.75
      High = 4118.5
      Low = 4116.75
      Close = 4117.75
      Volume = 450.0
      Time = 7/31/2022 8:45:00 PM }; { Open = 4118.0
                                       High = 4118.75
                                       Low = 4117.5
                                       Close = 4118.0
                                       Volume = 370.0
                                       Time = 7/31/2022 8:50:00 PM };
    { Open = 4118.0
      High = 4118.5
      Low = 4117.75
      Close = 4117.75
      Volume = 227.0
      Time = 7/31/2022 8:55:00 PM }; { Open = 4117.75
                                       High = 4118.0
                                       Low = 4116.5
                                       Close = 4117.25
                                       Volume = 403.0
                                       Time = 7/31/2022 9:00:00 PM };
    { Open = 4117.5
      High = 4118.25
      Low = 4115.25
      Close = 4116.5
      Volume = 705.0
      Time = 7/31/2022 9:05:00 PM }; { Open = 4116.25
                                       High = 4116.75
                                       Low = 4115.5
                                       Close = 4116.0
                                       Volume = 581.0
                                       Time = 7/31/2022 9:10:00 PM };
    { Open = 4115.75
      High = 4118.25
      Low = 4115.75
      Close = 4117.75
      Volume = 753.0
      Time = 7/31/2022 9:15:00 PM }; { Open = 4117.75
                                       High = 4119.5
                                       Low = 4117.25
                                       Close = 4119.25
                                       Volume = 876.0
                                       Time = 7/31/2022 9:20:00 PM };
    { Open = 4119.5
      High = 4120.0
      Low = 4117.75
      Close = 4118.0
      Volume = 1048.0
      Time = 7/31/2022 9:25:00 PM }; { Open = 4118.0
                                       High = 4119.0
                                       Low = 4117.75
                                       Close = 4118.75
                                       Volume = 317.0
                                       Time = 7/31/2022 9:30:00 PM };
    { Open = 4118.75
      High = 4120.5
      Low = 4116.75
      Close = 4120.0
      Volume = 1146.0
      Time = 7/31/2022 9:35:00 PM }; { Open = 4120.25
                                       High = 4121.25
                                       Low = 4119.25
                                       Close = 4120.25
                                       Volume = 1183.0
                                       Time = 7/31/2022 9:40:00 PM };
    { Open = 4120.0
      High = 4120.75
      Low = 4118.0
      Close = 4119.0
      Volume = 681.0
      Time = 7/31/2022 9:45:00 PM }; { Open = 4119.0
                                       High = 4119.0
                                       Low = 4116.75
                                       Close = 4117.0
                                       Volume = 706.0
                                       Time = 7/31/2022 9:50:00 PM };
    { Open = 4117.0
      High = 4119.25
      Low = 4116.5
      Close = 4119.0
      Volume = 1187.0
      Time = 7/31/2022 9:55:00 PM }; { Open = 4119.0
                                       High = 4120.5
                                       Low = 4118.25
                                       Close = 4119.5
                                       Volume = 693.0
                                       Time = 7/31/2022 10:00:00 PM };
    { Open = 4119.25
      High = 4119.75
      Low = 4117.75
      Close = 4118.0
      Volume = 415.0
      Time = 7/31/2022 10:05:00 PM }; { Open = 4118.0
                                        High = 4118.75
                                        Low = 4117.5
                                        Close = 4117.75
                                        Volume = 417.0
                                        Time = 7/31/2022 10:10:00 PM };
    { Open = 4118.0
      High = 4118.0
      Low = 4116.75
      Close = 4118.0
      Volume = 297.0
      Time = 7/31/2022 10:15:00 PM }; { Open = 4118.0
                                        High = 4118.0
                                        Low = 4117.0
                                        Close = 4117.5
                                        Volume = 390.0
                                        Time = 7/31/2022 10:20:00 PM };
    { Open = 4117.5
      High = 4118.5
      Low = 4117.5
      Close = 4118.0
      Volume = 257.0
      Time = 7/31/2022 10:25:00 PM }; { Open = 4118.0
                                        High = 4120.5
                                        Low = 4118.0
                                        Close = 4120.25
                                        Volume = 843.0
                                        Time = 7/31/2022 10:30:00 PM };
    { Open = 4120.25
      High = 4121.25
      Low = 4119.5
      Close = 4119.75
      Volume = 812.0
      Time = 7/31/2022 10:35:00 PM }; { Open = 4119.5
                                        High = 4119.75
                                        Low = 4118.25
                                        Close = 4118.75
                                        Volume = 492.0
                                        Time = 7/31/2022 10:40:00 PM };
    { Open = 4118.5
      High = 4119.25
      Low = 4118.25
      Close = 4118.75
      Volume = 160.0
      Time = 7/31/2022 10:45:00 PM }; { Open = 4118.75
                                        High = 4120.0
                                        Low = 4118.75
                                        Close = 4119.25
                                        Volume = 234.0
                                        Time = 7/31/2022 10:50:00 PM };
    { Open = 4119.25
      High = 4119.75
      Low = 4118.75
      Close = 4119.75
      Volume = 270.0
      Time = 7/31/2022 10:55:00 PM }; { Open = 4120.0
                                        High = 4121.0
                                        Low = 4119.75
                                        Close = 4120.0
                                        Volume = 592.0
                                        Time = 7/31/2022 11:00:00 PM };
    { Open = 4120.25
      High = 4120.25
      Low = 4118.75
      Close = 4118.75
      Volume = 468.0
      Time = 7/31/2022 11:05:00 PM }; { Open = 4118.5
                                        High = 4119.0
                                        Low = 4117.75
                                        Close = 4118.25
                                        Volume = 515.0
                                        Time = 7/31/2022 11:10:00 PM };
    { Open = 4118.25
      High = 4119.25
      Low = 4117.5
      Close = 4118.0
      Volume = 358.0
      Time = 7/31/2022 11:15:00 PM }; { Open = 4118.25
                                        High = 4118.5
                                        Low = 4115.5
                                        Close = 4115.5
                                        Volume = 1067.0
                                        Time = 7/31/2022 11:20:00 PM };
    { Open = 4115.75
      High = 4115.75
      Low = 4114.25
      Close = 4114.25
      Volume = 1014.0
      Time = 7/31/2022 11:25:00 PM }; { Open = 4114.5
                                        High = 4115.0
                                        Low = 4112.75
                                        Close = 4113.5
                                        Volume = 1321.0
                                        Time = 7/31/2022 11:30:00 PM };
    { Open = 4113.75
      High = 4114.25
      Low = 4113.0
      Close = 4114.25
      Volume = 573.0
      Time = 7/31/2022 11:35:00 PM }; { Open = 4114.25
                                        High = 4114.5
                                        Low = 4114.0
                                        Close = 4114.5
                                        Volume = 281.0
                                        Time = 7/31/2022 11:40:00 PM };
    { Open = 4114.5
      High = 4114.5
      Low = 4114.0
      Close = 4114.5
      Volume = 218.0
      Time = 7/31/2022 11:45:00 PM }; { Open = 4114.5
                                        High = 4114.5
                                        Low = 4113.0
                                        Close = 4113.25
                                        Volume = 441.0
                                        Time = 7/31/2022 11:50:00 PM };
    { Open = 4113.25
      High = 4114.25
      Low = 4113.0
      Close = 4114.0
      Volume = 326.0
      Time = 7/31/2022 11:55:00 PM }; { Open = 4113.75
                                        High = 4114.25
                                        Low = 4113.75
                                        Close = 4114.0
                                        Volume = 306.0
                                        Time = 8/1/2022 12:00:00 AM };
    { Open = 4114.0
      High = 4114.25
      Low = 4113.5
      Close = 4114.25
      Volume = 279.0
      Time = 8/1/2022 12:05:00 AM }; { Open = 4114.25
                                       High = 4114.25
                                       Low = 4113.5
                                       Close = 4113.5
                                       Volume = 330.0
                                       Time = 8/1/2022 12:10:00 AM };
    { Open = 4113.25
      High = 4114.5
      Low = 4113.25
      Close = 4114.25
      Volume = 242.0
      Time = 8/1/2022 12:15:00 AM }; { Open = 4114.25
                                       High = 4115.75
                                       Low = 4114.25
                                       Close = 4115.5
                                       Volume = 594.0
                                       Time = 8/1/2022 12:20:00 AM };
    { Open = 4115.5
      High = 4116.0
      Low = 4115.25
      Close = 4115.75
      Volume = 250.0
      Time = 8/1/2022 12:25:00 AM }; { Open = 4116.0
                                       High = 4116.25
                                       Low = 4115.5
                                       Close = 4116.25
                                       Volume = 195.0
                                       Time = 8/1/2022 12:30:00 AM };
    { Open = 4116.25
      High = 4116.5
      Low = 4115.0
      Close = 4116.0
      Volume = 529.0
      Time = 8/1/2022 12:35:00 AM }; { Open = 4116.0
                                       High = 4116.25
                                       Low = 4114.75
                                       Close = 4114.75
                                       Volume = 245.0
                                       Time = 8/1/2022 12:40:00 AM };
    { Open = 4114.75
      High = 4115.75
      Low = 4114.5
      Close = 4115.25
      Volume = 165.0
      Time = 8/1/2022 12:45:00 AM }; { Open = 4115.5
                                       High = 4115.5
                                       Low = 4114.75
                                       Close = 4115.0
                                       Volume = 103.0
                                       Time = 8/1/2022 12:50:00 AM };
    { Open = 4115.25
      High = 4115.5
      Low = 4114.5
      Close = 4115.5
      Volume = 211.0
      Time = 8/1/2022 12:55:00 AM }; { Open = 4115.75
                                       High = 4116.5
                                       Low = 4115.5
                                       Close = 4116.25
                                       Volume = 252.0
                                       Time = 8/1/2022 1:00:00 AM };
    { Open = 4116.25
      High = 4116.5
      Low = 4115.75
      Close = 4115.75
      Volume = 302.0
      Time = 8/1/2022 1:05:00 AM }; { Open = 4115.75
                                      High = 4116.25
                                      Low = 4115.25
                                      Close = 4115.75
                                      Volume = 212.0
                                      Time = 8/1/2022 1:10:00 AM };
    { Open = 4115.75
      High = 4115.75
      Low = 4113.75
      Close = 4114.25
      Volume = 381.0
      Time = 8/1/2022 1:15:00 AM }; { Open = 4114.25
                                      High = 4115.25
                                      Low = 4113.75
                                      Close = 4115.0
                                      Volume = 352.0
                                      Time = 8/1/2022 1:20:00 AM };
    { Open = 4115.0
      High = 4115.25
      Low = 4114.0
      Close = 4114.0
      Volume = 169.0
      Time = 8/1/2022 1:25:00 AM }; { Open = 4114.0
                                      High = 4115.5
                                      Low = 4114.0
                                      Close = 4115.5
                                      Volume = 163.0
                                      Time = 8/1/2022 1:30:00 AM };
    { Open = 4115.5
      High = 4117.0
      Low = 4115.0
      Close = 4115.0
      Volume = 657.0
      Time = 8/1/2022 1:35:00 AM }; { Open = 4114.75
                                      High = 4115.25
                                      Low = 4113.25
                                      Close = 4113.25
                                      Volume = 471.0
                                      Time = 8/1/2022 1:40:00 AM };
    { Open = 4113.25
      High = 4113.25
      Low = 4111.0
      Close = 4111.5
      Volume = 1062.0
      Time = 8/1/2022 1:45:00 AM }; { Open = 4111.5
                                      High = 4113.5
                                      Low = 4111.5
                                      Close = 4113.5
                                      Volume = 528.0
                                      Time = 8/1/2022 1:50:00 AM };
    { Open = 4113.5
      High = 4115.75
      Low = 4113.5
      Close = 4115.5
      Volume = 692.0
      Time = 8/1/2022 1:55:00 AM }; { Open = 4115.25
                                      High = 4116.75
                                      Low = 4115.25
                                      Close = 4116.75
                                      Volume = 603.0
                                      Time = 8/1/2022 2:00:00 AM };
    { Open = 4116.75
      High = 4118.25
      Low = 4116.5
      Close = 4117.75
      Volume = 1152.0
      Time = 8/1/2022 2:05:00 AM }; { Open = 4117.5
                                      High = 4118.0
                                      Low = 4116.5
                                      Close = 4116.75
                                      Volume = 632.0
                                      Time = 8/1/2022 2:10:00 AM };
    { Open = 4116.75
      High = 4118.0
      Low = 4116.0
      Close = 4118.0
      Volume = 422.0
      Time = 8/1/2022 2:15:00 AM }; { Open = 4118.0
                                      High = 4118.25
                                      Low = 4115.75
                                      Close = 4116.25
                                      Volume = 522.0
                                      Time = 8/1/2022 2:20:00 AM };
    { Open = 4116.25
      High = 4117.0
      Low = 4115.25
      Close = 4115.25
      Volume = 630.0
      Time = 8/1/2022 2:25:00 AM }; { Open = 4115.25
                                      High = 4117.0
                                      Low = 4114.5
                                      Close = 4116.75
                                      Volume = 523.0
                                      Time = 8/1/2022 2:30:00 AM };
    { Open = 4116.5
      High = 4116.5
      Low = 4114.5
      Close = 4115.25
      Volume = 610.0
      Time = 8/1/2022 2:35:00 AM }; { Open = 4115.75
                                      High = 4115.75
                                      Low = 4114.0
                                      Close = 4115.25
                                      Volume = 475.0
                                      Time = 8/1/2022 2:40:00 AM };
    { Open = 4115.25
      High = 4117.0
      Low = 4115.25
      Close = 4116.5
      Volume = 412.0
      Time = 8/1/2022 2:45:00 AM }; { Open = 4116.5
                                      High = 4118.25
                                      Low = ...
                                      Close = ...
                                      Volume = ...
                                      Time = ... }; ...|]
val d2Test: TsData.Bar = { Open = 4155.75
                           High = 4156.25
                           Low = 4155.25
                           Close = 4155.5
                           Volume = 263.0
                           Time = 8/24/2022 10:30:00 PM }
val d1Test: TsData.Bar = { Open = 4117.5
                           High = 4118.5
                           Low = 4116.75
                           Close = 4118.25
                           Volume = 638.0
                           Time = 7/31/2022 7:55:00 PM }
val mutable verbose: bool = false
type Parms =
  {
    CreateModel: (unit -> TorchSharp.Fun.IModel)
    DDQN: DDQN.DDQN
    LossFn: TorchSharp.Loss
    Opt: TorchSharp.torch.optim.Optimizer
    LearnEverySteps: int
    SyncEverySteps: int
    BatchSize: int
  }
  static member
    Default: modelFn: (unit -> TorchSharp.Fun.IModel) ->
               ddqn: DDQN.DDQN -> lr: float -> Parms
type RLState =
  {
    State: TorchSharp.torch.Tensor
    PrevState: TorchSharp.torch.Tensor
    Step: DDQN.Step
    InitialCash: float
    Stock: int
    CashOnHand: float
    LookBack: int64
    ExpBuff: DDQN.ExperienceBuffer
    S_reward: float
    S_gain: float
    Episode: int
  }
  static member Default: initExpRate: float -> initialCash: float -> RLState
  static member Reset: x: RLState -> RLState
type Market =
  { prices: TsData.Bar array }
  member IsDone: t: int -> bool
module Agent =
  val bar: env: Market -> t: int -> TsData.Bar option
  val avgPrice: bar: TsData.Bar -> float
  val buy: env: Market -> s: RLState -> RLState
  val sell: env: Market -> s: RLState -> RLState
  val doAction: 'a -> env: Market -> s: RLState -> act: int -> RLState
  val skipHead: TorchSharp.torch.TensorIndex = TorchSharp.torch+TensorIndex
  val getObservations: 'a -> env: Market -> s: RLState -> RLState
  val computeRewards:
    parms: Parms ->
      env: Market -> s: RLState -> action: int -> RLState * bool * float
  val agent: RL.Agent<Parms,Market,RLState> =
    { doAction = <fun:agent@161>
      getObservations = <fun:agent@162-1>
      computeRewards = <fun:agent@163-2> }
module Policy =
  val updateQ:
    parms: Parms ->
      td_estimate: TorchSharp.torch.Tensor ->
      td_target: TorchSharp.torch.Tensor -> float
  val learn: parms: Parms -> s: RLState -> RLState
  val syncModel: parms: Parms -> s: RLState -> unit
  val policy: parms: Parms -> RL.Policy<Parms,RLState>
module Test =
  val interimModel: string = "E:\s\tradestation\test_ddqn.bin"
  val saveInterim: parms: Parms -> unit
  val testMarket: unit -> Market
  val trainMarket: unit -> Market
  val evalModelTT:
    model: TorchSharp.Fun.IModel ->
      market: Market -> data: TsData.Bar[] -> refLen: int -> float
  val evalModel:
    name: string -> model: TorchSharp.Fun.IModel -> string * float * float
  val evalModelFile:
    parms: Parms -> modelFile: string -> string * float * float
  val copyModels: unit -> unit
  val evalModels: parms: Parms -> unit
  val runTest:
    parms: Parms -> (TorchSharp.Fun.IModel -> string * float * float)
  val clearModels: unit -> unit
val market: Market =
  { prices =
     [|{ Open = 3989.25
         High = 3990.25
         Low = 3985.25
         Close = 3985.25
         Volume = 1316.0
         Time = 7/21/2022 6:05:00 PM }; { Open = 3985.0
                                          High = 3985.75
                                          Low = 3982.75
                                          Close = 3984.25
                                          Volume = 979.0
                                          Time = 7/21/2022 6:10:00 PM };
       { Open = 3984.5
         High = 3985.0
         Low = 3982.75
         Close = 3984.5
         Volume = 637.0
         Time = 7/21/2022 6:15:00 PM }; { Open = 3984.25
                                          High = 3987.25
                                          Low = 3983.25
                                          Close = 3986.75
                                          Volume = 689.0
                                          Time = 7/21/2022 6:20:00 PM };
       { Open = 3987.0
         High = 3989.5
         Low = 3986.25
         Close = 3986.5
         Volume = 1259.0
         Time = 7/21/2022 6:25:00 PM }; { Open = 3986.75
                                          High = 3989.25
                                          Low = 3986.25
                                          Close = 3988.25
                                          Volume = 523.0
                                          Time = 7/21/2022 6:30:00 PM };
       { Open = 3988.0
         High = 3990.0
         Low = 3987.25
         Close = 3988.5
         Volume = 782.0
         Time = 7/21/2022 6:35:00 PM }; { Open = 3988.5
                                          High = 3988.5
                                          Low = 3986.0
                                          Close = 3986.75
                                          Volume = 797.0
                                          Time = 7/21/2022 6:40:00 PM };
       { Open = 3986.75
         High = 3987.25
         Low = 3986.0
         Close = 3986.75
         Volume = 311.0
         Time = 7/21/2022 6:45:00 PM }; { Open = 3986.75
                                          High = 3987.75
                                          Low = 3985.5
                                          Close = 3987.25
                                          Volume = 282.0
                                          Time = 7/21/2022 6:50:00 PM };
       { Open = 3987.25
         High = 3987.5
         Low = 3985.5
         Close = 3986.25
         Volume = 200.0
         Time = 7/21/2022 6:55:00 PM }; { Open = 3986.5
                                          High = 3987.25
                                          Low = 3986.0
                                          Close = 3986.5
                                          Volume = 237.0
                                          Time = 7/21/2022 7:00:00 PM };
       { Open = 3986.25
         High = 3986.75
         Low = 3985.0
         Close = 3986.25
         Volume = 346.0
         Time = 7/21/2022 7:05:00 PM }; { Open = 3986.0
                                          High = 3986.5
                                          Low = 3985.25
                                          Close = 3985.75
                                          Volume = 172.0
                                          Time = 7/21/2022 7:10:00 PM };
       { Open = 3985.75
         High = 3986.75
         Low = 3985.5
         Close = 3985.75
         Volume = 174.0
         Time = 7/21/2022 7:15:00 PM }; { Open = 3985.5
                                          High = 3986.25
                                          Low = 3984.25
                                          Close = 3985.0
                                          Volume = 563.0
                                          Time = 7/21/2022 7:20:00 PM };
       { Open = 3985.0
         High = 3985.5
         Low = 3984.0
         Close = 3985.25
         Volume = 197.0
         Time = 7/21/2022 7:25:00 PM }; { Open = 3985.25
                                          High = 3987.25
                                          Low = 3985.0
                                          Close = 3987.0
                                          Volume = 384.0
                                          Time = 7/21/2022 7:30:00 PM };
       { Open = 3986.75
         High = 3988.25
         Low = 3985.75
         Close = 3987.0
         Volume = 795.0
         Time = 7/21/2022 7:35:00 PM }; { Open = 3987.25
                                          High = 3987.25
                                          Low = 3985.0
                                          Close = 3985.5
                                          Volume = 384.0
                                          Time = 7/21/2022 7:40:00 PM };
       { Open = 3985.5
         High = 3987.0
         Low = 3985.5
         Close = 3986.5
         Volume = 184.0
         Time = 7/21/2022 7:45:00 PM }; { Open = 3986.5
                                          High = 3987.25
                                          Low = 3985.25
                                          Close = 3986.25
                                          Volume = 454.0
                                          Time = 7/21/2022 7:50:00 PM };
       { Open = 3986.25
         High = 3986.25
         Low = 3983.5
         Close = 3984.25
         Volume = 578.0
         Time = 7/21/2022 7:55:00 PM }; { Open = 3984.0
                                          High = 3985.0
                                          Low = 3983.5
                                          Close = 3984.25
                                          Volume = 488.0
                                          Time = 7/21/2022 8:00:00 PM };
       { Open = 3984.25
         High = 3984.25
         Low = 3980.5
         Close = 3983.0
         Volume = 1650.0
         Time = 7/21/2022 8:05:00 PM }; { Open = 3983.0
                                          High = 3983.5
                                          Low = 3981.75
                                          Close = 3981.75
                                          Volume = 668.0
                                          Time = 7/21/2022 8:10:00 PM };
       { Open = 3981.5
         High = 3982.5
         Low = 3980.5
         Close = 3982.25
         Volume = 618.0
         Time = 7/21/2022 8:15:00 PM }; { Open = 3982.0
                                          High = 3985.0
                                          Low = 3981.75
                                          Close = 3984.0
                                          Volume = 938.0
                                          Time = 7/21/2022 8:20:00 PM };
       { Open = 3984.25
         High = 3985.5
         Low = 3983.75
         Close = 3984.75
         Volume = 369.0
         Time = 7/21/2022 8:25:00 PM }; { Open = 3985.0
                                          High = 3986.0
                                          Low = 3983.75
                                          Close = 3983.75
                                          Volume = 631.0
                                          Time = 7/21/2022 8:30:00 PM };
       { Open = 3983.75
         High = 3986.0
         Low = 3983.75
         Close = 3984.75
         Volume = 385.0
         Time = 7/21/2022 8:35:00 PM }; { Open = 3984.75
                                          High = 3986.5
                                          Low = 3983.0
                                          Close = 3986.25
                                          Volume = 598.0
                                          Time = 7/21/2022 8:40:00 PM };
       { Open = 3986.25
         High = 3986.5
         Low = 3984.0
         Close = 3984.25
         Volume = 337.0
         Time = 7/21/2022 8:45:00 PM }; { Open = 3984.5
                                          High = 3985.75
                                          Low = 3984.0
                                          Close = 3985.25
                                          Volume = 325.0
                                          Time = 7/21/2022 8:50:00 PM };
       { Open = 3985.25
         High = 3986.75
         Low = 3985.25
         Close = 3986.25
         Volume = 397.0
         Time = 7/21/2022 8:55:00 PM }; { Open = 3986.0
                                          High = 3987.5
                                          Low = 3985.75
                                          Close = 3987.25
                                          Volume = 555.0
                                          Time = 7/21/2022 9:00:00 PM };
       { Open = 3987.25
         High = 3987.25
         Low = 3983.25
         Close = 3984.0
         Volume = 984.0
         Time = 7/21/2022 9:05:00 PM }; { Open = 3984.0
                                          High = 3985.0
                                          Low = 3983.5
                                          Close = 3984.0
                                          Volume = 525.0
                                          Time = 7/21/2022 9:10:00 PM };
       { Open = 3983.75
         High = 3984.0
         Low = 3982.5
         Close = 3983.75
         Volume = 464.0
         Time = 7/21/2022 9:15:00 PM }; { Open = 3983.5
                                          High = 3984.0
                                          Low = 3981.25
                                          Close = 3982.75
                                          Volume = 687.0
                                          Time = 7/21/2022 9:20:00 PM };
       { Open = 3982.5
         High = 3984.25
         Low = 3982.25
         Close = 3983.25
         Volume = 475.0
         Time = 7/21/2022 9:25:00 PM }; { Open = 3983.25
                                          High = 3985.75
                                          Low = 3983.25
                                          Close = 3984.75
                                          Volume = 570.0
                                          Time = 7/21/2022 9:30:00 PM };
       { Open = 3984.75
         High = 3987.0
         Low = 3984.75
         Close = 3986.75
         Volume = 744.0
         Time = 7/21/2022 9:35:00 PM }; { Open = 3987.0
                                          High = 3989.25
                                          Low = 3986.0
                                          Close = 3988.5
                                          Volume = 1144.0
                                          Time = 7/21/2022 9:40:00 PM };
       { Open = 3988.5
         High = 3991.5
         Low = 3988.25
         Close = 3990.0
         Volume = 1503.0
         Time = 7/21/2022 9:45:00 PM }; { Open = 3989.75
                                          High = 3990.0
                                          Low = 3987.0
                                          Close = 3987.0
                                          Volume = 965.0
                                          Time = 7/21/2022 9:50:00 PM };
       { Open = 3987.25
         High = 3988.5
         Low = 3986.0
         Close = 3987.5
         Volume = 607.0
         Time = 7/21/2022 9:55:00 PM }; { Open = 3987.25
                                          High = 3988.5
                                          Low = 3986.75
                                          Close = 3988.25
                                          Volume = 475.0
                                          Time = 7/21/2022 10:00:00 PM };
       { Open = 3988.25
         High = 3989.25
         Low = 3987.25
         Close = 3988.75
         Volume = 583.0
         Time = 7/21/2022 10:05:00 PM }; { Open = 3988.5
                                           High = 3989.0
                                           Low = 3986.75
                                           Close = 3987.0
                                           Volume = 322.0
                                           Time = 7/21/2022 10:10:00 PM };
       { Open = 3987.0
         High = 3987.0
         Low = 3985.25
         Close = 3985.25
         Volume = 589.0
         Time = 7/21/2022 10:15:00 PM }; { Open = 3985.25
                                           High = 3986.25
                                           Low = 3984.0
                                           Close = 3985.75
                                           Volume = 719.0
                                           Time = 7/21/2022 10:20:00 PM };
       { Open = 3986.0
         High = 3986.5
         Low = 3984.75
         Close = 3984.75
         Volume = 435.0
         Time = 7/21/2022 10:25:00 PM }; { Open = 3984.75
                                           High = 3985.5
                                           Low = 3983.5
                                           Close = 3984.25
                                           Volume = 555.0
                                           Time = 7/21/2022 10:30:00 PM };
       { Open = 3984.0
         High = 3985.0
         Low = 3983.75
         Close = 3983.75
         Volume = 320.0
         Time = 7/21/2022 10:35:00 PM }; { Open = 3983.75
                                           High = 3983.75
                                           Low = 3982.0
                                           Close = 3982.25
                                           Volume = 652.0
                                           Time = 7/21/2022 10:40:00 PM };
       { Open = 3982.5
         High = 3984.5
         Low = 3982.25
         Close = 3984.25
         Volume = 494.0
         Time = 7/21/2022 10:45:00 PM }; { Open = 3984.5
                                           High = 3984.75
                                           Low = 3983.25
                                           Close = 3983.75
                                           Volume = 422.0
                                           Time = 7/21/2022 10:50:00 PM };
       { Open = 3983.5
         High = 3985.5
         Low = 3983.0
         Close = 3984.25
         Volume = 777.0
         Time = 7/21/2022 10:55:00 PM }; { Open = 3984.25
                                           High = 3984.5
                                           Low = 3983.0
                                           Close = 3983.5
                                           Volume = 341.0
                                           Time = 7/21/2022 11:00:00 PM };
       { Open = 3983.5
         High = 3985.25
         Low = 3983.25
         Close = 3984.5
         Volume = 358.0
         Time = 7/21/2022 11:05:00 PM }; { Open = 3984.25
                                           High = 3984.75
                                           Low = 3983.5
                                           Close = 3984.5
                                           Volume = 263.0
                                           Time = 7/21/2022 11:10:00 PM };
       { Open = 3984.5
         High = 3985.5
         Low = 3984.25
         Close = 3985.0
         Volume = 275.0
         Time = 7/21/2022 11:15:00 PM }; { Open = 3985.0
                                           High = 3986.0
                                           Low = 3983.75
                                           Close = 3983.75
                                           Volume = 368.0
                                           Time = 7/21/2022 11:20:00 PM };
       { Open = 3983.5
         High = 3984.25
         Low = 3982.25
         Close = 3983.0
         Volume = 513.0
         Time = 7/21/2022 11:25:00 PM }; { Open = 3983.0
                                           High = 3985.0
                                           Low = 3982.75
                                           Close = 3984.75
                                           Volume = 335.0
                                           Time = 7/21/2022 11:30:00 PM };
       { Open = 3984.75
         High = 3986.0
         Low = 3984.75
         Close = 3985.75
         Volume = 423.0
         Time = 7/21/2022 11:35:00 PM }; { Open = 3985.5
                                           High = 3986.0
                                           Low = 3984.5
                                           Close = 3985.25
                                           Volume = 302.0
                                           Time = 7/21/2022 11:40:00 PM };
       { Open = 3985.0
         High = 3986.0
         Low = 3984.75
         Close = 3986.0
         Volume = 255.0
         Time = 7/21/2022 11:45:00 PM }; { Open = 3986.0
                                           High = 3986.75
                                           Low = 3985.5
                                           Close = 3986.0
                                           Volume = 402.0
                                           Time = 7/21/2022 11:50:00 PM };
       { Open = 3985.75
         High = 3986.0
         Low = 3984.75
         Close = 3985.0
         Volume = 279.0
         Time = 7/21/2022 11:55:00 PM }; { Open = 3985.25
                                           High = 3986.5
                                           Low = 3984.75
                                           Close = 3986.25
                                           Volume = 259.0
                                           Time = 7/22/2022 12:00:00 AM };
       { Open = 3986.5
         High = 3987.25
         Low = 3986.25
         Close = 3987.25
         Volume = 332.0
         Time = 7/22/2022 12:05:00 AM }; { Open = 3987.0
                                           High = 3987.5
                                           Low = 3986.5
                                           Close = 3987.25
                                           Volume = 190.0
                                           Time = 7/22/2022 12:10:00 AM };
       { Open = 3987.5
         High = 3988.0
         Low = 3986.5
         Close = 3986.5
         Volume = 393.0
         Time = 7/22/2022 12:15:00 AM }; { Open = 3986.5
                                           High = 3986.75
                                           Low = 3985.75
                                           Close = 3986.5
                                           Volume = 177.0
                                           Time = 7/22/2022 12:20:00 AM };
       { Open = 3986.75
         High = 3987.5
         Low = 3986.5
         Close = 3986.75
         Volume = 256.0
         Time = 7/22/2022 12:25:00 AM }; { Open = 3987.0
                                           High = 3987.0
                                           Low = 3985.75
                                           Close = 3986.0
                                           Volume = 171.0
                                           Time = 7/22/2022 12:30:00 AM };
       { Open = 3986.0
         High = 3986.5
         Low = 3984.75
         Close = 3984.75
         Volume = 332.0
         Time = 7/22/2022 12:35:00 AM }; { Open = 3985.0
                                           High = 3986.25
                                           Low = 3984.75
                                           Close = 3985.5
                                           Volume = 154.0
                                           Time = 7/22/2022 12:40:00 AM };
       { Open = 3985.5
         High = 3985.75
         Low = 3984.75
         Close = 3985.0
         Volume = 172.0
         Time = 7/22/2022 12:45:00 AM }; { Open = 3985.25
                                           High = 3985.75
                                           Low = 3984.75
                                           Close = 3985.0
                                           Volume = 227.0
                                           Time = 7/22/2022 12:50:00 AM };
       { Open = 3985.0
         High = 3985.25
         Low = 3984.5
         Close = 3984.75
         Volume = 168.0
         Time = 7/22/2022 12:55:00 AM }; { Open = 3985.0
                                           High = ...
                                           Low = ...
                                           Close = ...
                                           Volume = ...
                                           Time = ... }; ...|] }
val runEpisode:
  parms: Parms ->
    p: RL.Policy<Parms,RLState> * s: RLState ->
      RL.Policy<Parms,RLState> * RLState
val run:
  parms: Parms ->
    p: RL.Policy<Parms,RLState> * s: RLState ->
      RL.Policy<Parms,RLState> * RLState
val resetRun: parms: Parms -> RL.Policy<Parms,RLState> * RLState
val mutable _ps: RL.Policy<Parms,RLState> * RLState = <null>
val startResetRun: parms: Parms -> unit
val startReRun: parms: Parms -> unit
val parms1: unit -> Parms

> 
val p1: Parms =Binding session to 'C:/Users/Faisa/.nuget/packages/fsharpx.collections/3.0.1/lib/netstandard2.0/FSharpx.Collections.dll'...
 { CreateModel = <fun:createModel@373>
                  DDQN = { Model = { Target = FSI_0007+createModel@373-55
                                     Online = FSI_0007+createModel@373-55 }
                           Gamma = 0.9998999834f
                           Exploration = { Decay = 0.9995
                                           Min = 0.01 }
                           Actions = 2
                           Device = cuda }
                  LossFn = TorchSharp.Loss
                  Opt = TorchSharp.Modules.Adam
                  LearnEverySteps = 3
                  SyncEverySteps = 15000
                  BatchSize = 300 }
val it: unit = ()

> Run: 0, R:0, E:0.010; Cash:2436.50; Stock:225; Gain:-0.014482; Experienced:50000
model: current, Adg. Gain -  Test: 0.32081472, Train: -0.10765225
Run: 1, R:-0, E:0.010; Cash:1053005.25; Stock:0; Gain:0.053005; Experienced:50000
model: current, Adg. Gain -  Test: 0.30765168, Train: 0.013942624999999998
Run: 2, R:0, E:0.010; Cash:1002.62; Stock:216; Gain:-0.055239; Experienced:50000
model: current, Adg. Gain -  Test: 0.19185839999999998, Train: -0.007993625
Run: 3, R:0, E:0.010; Cash:3536.62; Stock:314; Gain:0.375481; Experienced:50000
model: current, Adg. Gain -  Test: -0.10790208, Train: 0.24232362499999996
Run: 4, R:0, E:0.010; Cash:2439.25; Stock:300; Gain:0.313214; Experienced:50000
model: current, Adg. Gain -  Test: -0.07748496, Train: 0.17728275
Run: 5, R:0, E:0.010; Cash:2990.38; Stock:309; Gain:0.353089; Experienced:50000
model: current, Adg. Gain -  Test: 0.19987056, Train: -0.001505125
Run: 6, R:-0, E:0.010; Cash:1285803.75; Stock:0; Gain:0.285804; Experienced:50000
model: current, Adg. Gain -  Test: 0.043894079999999995, Train: 0.110911125
Run: 7, R:0, E:0.010; Cash:899.50; Stock:262; Gain:0.145643; Experienced:50000
model: current, Adg. Gain -  Test: -0.094536, Train: 0.166035875
Run: 8, R:0, E:0.010; Cash:271.25; Stock:270; Gain:0.179969; Experienced:50000
model: current, Adg. Gain -  Test: 0.12730608, Train: 0.238950875
Run: 9, R:0, E:0.010; Cash:3348.38; Stock:300; Gain:0.314123; Experienced:50000
model: current, Adg. Gain -  Test: 0.23740848000000003, Train: 0.297948375
Run: 10, R:0, E:0.010; Cash:1229.88; Stock:262; Gain:0.145973; Experienced:50000
model: current, Adg. Gain -  Test: -0.06306192, Train: 0.10938725
Run: 11, R:0, E:0.010; Cash:52.75; Stock:298; Gain:0.302089; Experienced:50000
model: current, Adg. Gain -  Test: 0.04427712, Train: 0.212688875
Run: 12, R:0, E:0.010; Cash:3934.62; Stock:249; Gain:0.091878; Experienced:50000
model: current, Adg. Gain -  Test: -0.05147568, Train: 0.094197875
Run: 13, R:0, E:0.010; Cash:2783.88; Stock:303; Gain:0.326667; Experienced:50000
model: current, Adg. Gain -  Test: -0.0036172799999999996, Train: 0.25576125
Run: 14, R:0, E:0.010; Cash:331.25; Stock:297; Gain:0.297998; Experienced:50000
model: current, Adg. Gain -  Test: -0.06450768, Train: 0.14140925
Run: 15, R:0, E:0.010; Cash:1628.50; Stock:251; Gain:0.098310; Experienced:50000
model: current, Adg. Gain -  Test: -0.13275792, Train: 0.07040375
Run: 16, R:-0, E:0.010; Cash:1120355.00; Stock:0; Gain:0.120355; Experienced:50000
model: current, Adg. Gain -  Test: -0.12071376, Train: 0.150888875
Run: 17, R:0, E:0.010; Cash:433.25; Stock:247; Gain:0.079638; Experienced:50000
model: current, Adg. Gain -  Test: 0.23951952000000004, Train: 0.2840675
Run: 18, R:0, E:0.010; Cash:774.00; Stock:251; Gain:0.097456; Experienced:50000
model: current, Adg. Gain -  Test: -0.021045599999999998, Train: 0.213496875
Run: 19, R:0, E:0.010; Cash:1651.75; Stock:278; Gain:0.216303; Experienced:50000
model: current, Adg. Gain -  Test: 0.07835328000000001, Train: 0.175462125
Run: 20, R:-0, E:0.010; Cash:1212291.12; Stock:0; Gain:0.212291; Experienced:50000
model: current, Adg. Gain -  Test: -0.06022944000000001, Train: 0.088838625
Run: 21, R:-0, E:0.010; Cash:1133973.25; Stock:0; Gain:0.133973; Experienced:50000
model: current, Adg. Gain -  Test: -0.13176144, Train: 0.199819125
Run: 22, R:-0, E:0.010; Cash:1194910.38; Stock:0; Gain:0.194910; Experienced:50000
model: current, Adg. Gain -  Test: 0.12999455999999998, Train: 0.253589875
Run: 23, R:0, E:0.010; Cash:2500.50; Stock:236; Gain:0.033643; Experienced:50000
model: current, Adg. Gain -  Test: -0.00253728, Train: 0.1314595
Run: 24, R:0, E:0.010; Cash:456.25; Stock:258; Gain:0.127723; Experienced:50000
model: current, Adg. Gain -  Test: -0.08926992, Train: 0.138952625
Run: 25, R:-0, E:0.010; Cash:1278734.38; Stock:0; Gain:0.278734; Experienced:50000
model: current, Adg. Gain -  Test: 0.05083920000000001, Train: 0.1731305
Run: 26, R:0, E:0.010; Cash:843.12; Stock:297; Gain:0.298510; Experienced:50000
model: current, Adg. Gain -  Test: 0.01612512, Train: 0.160763625
Run: 27, R:0, E:0.010; Cash:1608.12; Stock:261; Gain:0.141982; Experienced:50000
model: current, Adg. Gain -  Test: 0.084096, Train: 0.1621735
Run: 28, R:0, E:0.010; Cash:3319.00; Stock:276; Gain:0.209232; Experienced:50000
model: current, Adg. Gain -  Test: -0.01943424, Train: 0.23800199999999996
Run: 29, R:0, E:0.010; Cash:3973.12; Stock:257; Gain:0.126870; Experienced:50000
model: current, Adg. Gain -  Test: 0.057434400000000004, Train: 0.161912125
Run: 30, R:0, E:0.010; Cash:1631.88; Stock:266; Gain:0.163852; Experienced:50000
model: current, Adg. Gain -  Test: -0.01449504, Train: 0.1675245
Run: 31, R:0, E:0.010; Cash:1825.25; Stock:297; Gain:0.299492; Experienced:50000
model: current, Adg. Gain -  Test: -0.012132, Train: 0.12426212499999999
Run: 32, R:0, E:0.010; Cash:2172.12; Stock:261; Gain:0.142546; Experienced:50000
model: current, Adg. Gain -  Test: 0.055673280000000006, Train: 0.15301075
Run: 33, R:0, E:0.010; Cash:2047.12; Stock:267; Gain:0.168637; Experienced:50000
model: current, Adg. Gain -  Test: -0.053781119999999995, Train: 0.111283
Run: 34, R:0, E:0.010; Cash:1667.88; Stock:275; Gain:0.203212; Experienced:50000
model: current, Adg. Gain -  Test: -0.09389664, Train: 0.09551575
Run: 35, R:0, E:0.010; Cash:943.00; Stock:284; Gain:0.241810; Experienced:50000
model: current, Adg. Gain -  Test: 0.069516, Train: 0.12502925
Run: 36, R:0, E:0.010; Cash:886.50; Stock:280; Gain:0.224276; Experienced:50000
model: current, Adg. Gain -  Test: -0.09093168, Train: 0.132858125
Run: 37, R:0, E:0.010; Cash:463.62; Stock:233; Gain:0.018499; Experienced:50000
model: current, Adg. Gain -  Test: 0.027204479999999996, Train: 0.125320875
Run: 38, R:0, E:0.010; Cash:1163.00; Stock:293; Gain:0.281353; Experienced:50000
model: current, Adg. Gain -  Test: 0.16515648, Train: 0.099821375
Run: 39, R:0, E:0.010; Cash:24.25; Stock:270; Gain:0.179722; Experienced:50000
model: current, Adg. Gain -  Test: 0.02872368, Train: 0.118461875
Run: 40, R:0, E:0.010; Cash:631.38; Stock:276; Gain:0.206544; Experienced:50000
model: current, Adg. Gain -  Test: 0.061616159999999996, Train: 0.11441274999999998
Run: 41, R:0, E:0.010; Cash:1754.00; Stock:243; Gain:0.063482; Experienced:50000
model: current, Adg. Gain -  Test: 0.06642864, Train: 0.093904375
Run: 42, R:0, E:0.010; Cash:411.00; Stock:272; Gain:0.188847; Experienced:50000
model: current, Adg. Gain -  Test: 0.11905776000000001, Train: 0.126852
Run: 43, R:0, E:0.010; Cash:1063.88; Stock:284; Gain:0.241931; Experienced:50000
model: current, Adg. Gain -  Test: -0.043515359999999996, Train: 0.10480125
Run: 44, R:0, E:0.010; Cash:2128.88; Stock:235; Gain:0.028903; Experienced:50000
model: current, Adg. Gain -  Test: -0.05351328, Train: 0.12340225
Run: 45, R:0, E:0.010; Cash:2518.50; Stock:295; Gain:0.291447; Experienced:50000
model: current, Adg. Gain -  Test: 0.10028448, Train: 0.07155425
Run: 46, R:0, E:0.010; Cash:81.12; Stock:291; Gain:0.271533; Experienced:50000
model: current, Adg. Gain -  Test: 0.16367328, Train: 0.133850125
Run: 47, R:0, E:0.010; Cash:462.38; Stock:270; Gain:0.180160; Experienced:50000
model: current, Adg. Gain -  Test: 0.16562304, Train: 0.07956125
Run: 48, R:0, E:0.010; Cash:394.88; Stock:238; Gain:0.040276; Experienced:50000
model: current, Adg. Gain -  Test: 0.19841472, Train: 0.10231175
Run: 49, R:-0, E:0.010; Cash:1209422.25; Stock:0; Gain:0.209422; Experienced:50000
model: current, Adg. Gain -  Test: 0.18325727999999997, Train: 0.133697625
Run: 50, R:0, E:0.010; Cash:2678.38; Stock:274; Gain:0.199853; Experienced:50000
model: current, Adg. Gain -  Test: 0.04153824, Train: 0.092386125
Run: 51, R:0, E:0.010; Cash:1344.00; Stock:295; Gain:0.290273; Experienced:50000
model: current, Adg. Gain -  Test: 0.10207728, Train: 0.1070985
Run: 52, R:0, E:0.010; Cash:1434.12; Stock:245; Gain:0.071900; Experienced:50000
model: current, Adg. Gain -  Test: 0.08348976, Train: 0.184061375
Run: 53, R:0, E:0.010; Cash:307.00; Stock:251; Gain:0.096989; Experienced:50000
model: current, Adg. Gain -  Test: 0.16259472, Train: 0.066092625
Run: 54, R:0, E:0.010; Cash:1853.75; Stock:289; Gain:0.264567; Experienced:50000
model: current, Adg. Gain -  Test: 0.18736560000000002, Train: 0.0753165
Run: 55, R:0, E:0.010; Cash:1658.75; Stock:253; Gain:0.107079; Experienced:50000
model: current, Adg. Gain -  Test: -0.0172368, Train: 0.076051625
Run: 56, R:0, E:0.010; Cash:3164.25; Stock:298; Gain:0.305201; Experienced:50000
model: current, Adg. Gain -  Test: 0.017472960000000003, Train: 0.06775825
Run: 57, R:0, E:0.010; Cash:2476.62; Stock:259; Gain:0.134112; Experienced:50000
model: current, Adg. Gain -  Test: 0.27269424000000003, Train: 0.06022949999999999
Run: 58, R:0, E:0.010; Cash:503.25; Stock:275; Gain:0.202047; Experienced:50000
model: current, Adg. Gain -  Test: 0.18308736000000003, Train: 0.045378125
Run: 59, R:-0, E:0.010; Cash:1111972.75; Stock:0; Gain:0.111973; Experienced:50000
model: current, Adg. Gain -  Test: 0.06375744000000001, Train: 0.05106525
Run: 60, R:0, E:0.010; Cash:1988.88; Stock:235; Gain:0.028763; Experienced:50000
model: current, Adg. Gain -  Test: 0.053408159999999996, Train: 0.091257625
Run: 61, R:-0, E:0.010; Cash:1224052.62; Stock:0; Gain:0.224053; Experienced:50000
model: current, Adg. Gain -  Test: 0.066168, Train: 0.082595375
Run: 62, R:0, E:0.010; Cash:1909.62; Stock:254; Gain:0.111699; Experienced:50000
model: current, Adg. Gain -  Test: -0.06239088, Train: 0.12764475
Run: 63, R:0, E:0.010; Cash:801.38; Stock:259; Gain:0.132437; Experienced:50000
model: current, Adg. Gain -  Test: 0.11702736, Train: 0.105620125
Run: 64, R:0, E:0.010; Cash:4354.88; Stock:269; Gain:0.179683; Experienced:50000
model: current, Adg. Gain -  Test: 0.0324072, Train: 0.081951875
Run: 65, R:0, E:0.010; Cash:353.12; Stock:255; Gain:0.114512; Experienced:50000
model: current, Adg. Gain -  Test: 0.11976336, Train: 0.1075915
Run: 66, R:0, E:0.010; Cash:3818.38; Stock:234; Gain:0.026223; Experienced:50000
model: current, Adg. Gain -  Test: 0.14931648, Train: 0.103307875
Run: 67, R:0, E:0.010; Cash:456.00; Stock:273; Gain:0.193261; Experienced:50000
model: current, Adg. Gain -  Test: 0.11383344, Train: -0.005825625
Run: 68, R:-0, E:0.010; Cash:1052737.25; Stock:0; Gain:0.052737; Experienced:50000
model: current, Adg. Gain -  Test: 0.09904032, Train: 0.11012875000000001
Run: 69, R:0, E:0.010; Cash:1344.38; Stock:224; Gain:-0.019944; Experienced:50000
model: current, Adg. Gain -  Test: 0.18552960000000002, Train: 0.15449825
Run: 70, R:0, E:0.010; Cash:743.12; Stock:294; Gain:0.285303; Experienced:50000
model: current, Adg. Gain -  Test: 0.16405776, Train: 0.15414925
Run: 71, R:0, E:0.010; Cash:2420.75; Stock:295; Gain:0.291349; Experienced:50000
model: current, Adg. Gain -  Test: -0.02142576, Train: 0.151132
Run: 72, R:0, E:0.010; Cash:1123.75; Stock:266; Gain:0.163344; Experienced:50000
model: current, Adg. Gain -  Test: 0.34571519999999994, Train: 0.115856375
Run: 73, R:0, E:0.010; Cash:2909.75; Stock:264; Gain:0.156392; Experienced:50000
model: current, Adg. Gain -  Test: 0.06883631999999999, Train: 0.05789212500000001

"""

#r "nuget: Plotly.NET"
open System.IO
open Plotly.NET
open System

let lines() = 
    seq {
        use sr = new StringReader(s)
        let mutable ln = sr.ReadLine()
        while ln <> null do
            if String.IsNullOrWhiteSpace ln |> not then 
                yield ln
            ln <- sr.ReadLine()
    }

lines() |> Seq.toArray

let gains = 
    lines() 
    |> Seq.filter (fun l -> l.StartsWith("Run")) 
    |> Seq.map (fun x->x.Split([|':';' ';',';';'|],StringSplitOptions.RemoveEmptyEntries))
    |> Seq.map (fun xs -> {|Run=int xs.[1]; Gain= float xs.[11]|})
    |> Seq.toArray

let evals = 
    lines() 
    |> Seq.filter (fun l -> l.StartsWith("model")) 
    |> Seq.map (fun x->x.Split([|':';' ';',';';'|],StringSplitOptions.RemoveEmptyEntries))
    |> Seq.map (fun xs -> {|Test=float xs.[6]; Train= float xs.[8]|})
    |> Seq.toArray

gains |> Seq.map (fun g -> g.Run,g.Gain) |> Chart.Line |> Chart.show
evals |> Seq.map (fun g -> g.Train,g.Test) |> Chart.Line |> Chart.show
let testEvals = evals |> Seq.map (fun g -> g.Test)
let trainEvals = evals |> Seq.map (fun g->g.Train)
Chart.Histogram2D(trainEvals,testEvals)  |> Chart.show 






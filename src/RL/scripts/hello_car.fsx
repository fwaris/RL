#load "packages.fsx"
open System
open AirSimCar
open OpenCvSharp
open System.Threading.Tasks
open TorchSharp

let inline exec (t:Task<_>) = t |> Async.AwaitTask |> Async.RunSynchronously

module Image =
    open OpenCvSharp
    let clearWin() = Window.DestroyAllWindows()

    let showPng (bytes:byte[]) =
        async {
            let im2 = Cv2.ImDecode(bytes,ImreadModes.Unchanged)
            try
                Window.ShowImages(im2)
                im2.Release()
            with ex -> ()
        }
        |> Async.Start

    let showGray (w,h,data:float32[]) =
        async {
            let im2 = new Mat([h;w],MatType.CV_32FC1,data)
            try
                Window.ShowImages(im2)
                im2.Release()
            with ex -> ()
        }
        |> Async.Start

    let showGray2 (w,h,data:byte[]) =
        async {
            let im2 = new Mat([h;w],MatType.CV_8U,data)
            try
                Window.ShowImages(im2)
                im2.Release()
            with ex -> ()
        }
        |> Async.Start

let c1 = new CarClient(AirSimCar.Defaults.options)

c1.Connect(AirSimCar.Defaults.address,AirSimCar.Defaults.port)
c1.getServerVersion() |> exec
c1.enableApiControl(true) |> exec
c1.isApiControlEnabled() |> exec
c1.armDisarm(true) |> exec
c1.reset() |> exec
c1.setCarControls({CarControls.Default with throttle = 1.0 }) |> exec
let cs = c1.getCarState() |> exec
let ci = c1.simGetCollisionInfo() |> exec

c1.enableApiControl(false) |> exec 

let obs = c1.simListSceneObjects() |> exec 
obs |> List.choose(fun x->
    let y= x.IndexOf("_")
    if y > 0 then
        x.Substring(0,y) |> Some
    else
        None)
    |> List.countBy (fun x -> x)
    |> List.iter (printfn "%A")

let carId = c1.simListSceneObjects(name_regex="P.*Car") |> exec |> List.head
let p1 = c1.simGetObjectPose(carId) |> exec
let p2 = {p1 with position = {p1.position with x_val=100.0}; orientation={p1.orientation with z_val=0.11}}
let p3 = {p1 with orientation={p1.orientation with z_val= -0.03}}
c1.simSetObjectPose(carId,p3,true) |> exec

//rotate car
for z in -1.0 .. 0.1 .. 1.0 do
    let pz = {p1 with orientation={p1.orientation with z_val= z}}
    let isSet = c1.simSetObjectPose(carId,pz,true) |> exec
    printfn $"z={z}, isSet={isSet}"
    Threading.Thread.Sleep(1000)


let pr = CarEnvironment.randPose()
c1.simSetObjectPose(CarEnvironment.carId,pr,true) |> exec
let pr2 = {pr with orientation = {pr.orientation with z_val= 1.0; w_val=0.0}}
c1.simSetObjectPose(CarEnvironment.carId,pr2,true) |> exec




let km1 = c1.getCarState() |> exec
km1.kinematics_estimated.position
km1.kinematics_estimated.orientation

let gt = c1.simGetGroundTruthKinematics() |> exec
let km2 = {gt with position={gt.position with x_val=5.0}}
c1.simSetKinematics(km2) |> exec

c1.Disconnect()


let im1 = c1.simGetImage("1",ImageType.Scene) |> exec
let im2 = c1.simGetImage("0",ImageType.Scene) |> exec

let im5s = c1.simGetImages([|{camera_name="0"; image_type=ImageType.DepthVis; pixels_as_float=true; compress=false}|]) |> exec
let im3s = c1.simGetImages([|{camera_name="1"; image_type=ImageType.DepthPerspective; pixels_as_float=false; compress=false}|]) |> exec
let im4s = c1.simGetImages([|{camera_name="0"; image_type=ImageType.DepthPerspective; pixels_as_float=true; compress=false}|]) |> exec
let im4 = im4s.[0]

let txIm4 = CarEnvironment.transformImage im4
txIm4.data<float32>().ToArray() |> Array.max
let m = new Mat(84,84, MatType.CV_16S, txIm4.data<float32>().ToArray())
Window.ShowImages m

Image.showPng(im1) 
Image.showPng(im2)
let im3 = im3s.[0]
Image.showGray2(im3.width,im3.height,im3.image_data_uint8)
Image.showPng(im3.image_data_uint8)
c1.enableApiControl(false)

let im5 = im5s.[0]        
Image.showGray(im5.width,im5.height,im5.image_data_float)


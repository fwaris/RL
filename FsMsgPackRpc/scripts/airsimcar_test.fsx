#load "packages.fsx"
open System
open AirSimCar

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
c1.getServerVersion() |> Async.AwaitTask |> Async.RunSynchronously
c1.enableApiControl(true) |> Async.AwaitTask |> Async.RunSynchronously
c1.isApiControlEnabled() |> Async.AwaitTask |> Async.RunSynchronously
c1.armDisarm(true) |> Async.AwaitTask |> Async.RunSynchronously
c1.reset() |> Async.AwaitTask |> Async.RunSynchronously
c1.setCarControls({CarControls.Default with throttle = 1.0 }) |> Async.AwaitTask |> Async.RunSynchronously
let cs = c1.getCarState() |> Async.AwaitTask |> Async.RunSynchronously
let ci = c1.simGetCollisionInfo() |> Async.AwaitTask |> Async.RunSynchronously
c1.Disconnect()


let im1 = c1.simGetImage("1",ImageType.Scene) |> Async.AwaitTask |> Async.RunSynchronously
let im2 = c1.simGetImage("0",ImageType.Scene) |> Async.AwaitTask |> Async.RunSynchronously

let im3s = c1.simGetImages([|{camera_name="1"; image_type=ImageType.DepthPerspective; pixels_as_float=false; compress=false}|]) |> Async.AwaitTask |> Async.RunSynchronously
let im4s = c1.simGetImages([|{camera_name="1"; image_type=ImageType.DepthPerspective; pixels_as_float=true; compress=false}|]) |> Async.AwaitTask |> Async.RunSynchronously
let im5s = c1.simGetImages([|{camera_name="1"; image_type=ImageType.DepthVis; pixels_as_float=true; compress=false}|]) |> Async.AwaitTask |> Async.RunSynchronously

Image.showPng(im1) 
Image.showPng(im2)
let im3 = im3s.[0]
Image.showGray2(im3.width,im3.height,im3.image_data_uint8)
Image.showPng(im3.image_data_uint8)
c1.enableApiControl(false)
let im4 = im4s.[0]
Image.showGray(im4.width,im4.height,im4.image_data_float)

let im5 = im5s.[0]        
Image.showGray(im5.width,im5.height,im5.image_data_float)


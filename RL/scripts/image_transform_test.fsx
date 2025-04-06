#load "packages.fsx"
open System
open AirSimCar
open TorchSharp

let c = new CarClient(AirSimCar.Defaults.options)
c.Connect(AirSimCar.Defaults.address,AirSimCar.Defaults.port)
c.reset()

c.setCarControls({CarControls.Default with throttle=1.0}) |> Async.AwaitTask |> Async.RunSynchronously

let img1 = c.simGetImages(CarEnvironment.imageRequest) |> Async.AwaitTask |> Async.RunSynchronously
let resp = img1.[0]
resp.image_data_float

let t1 = torch.tensor resp.image_data_float
let t2 = 255.f.ToScalar() / torch.maximum(torch.ones_like t1, t1)
t2.data<float32>().ToArray()
let t3 = t2.reshape(1,resp.height,resp.width)
let t4 = torchvision.transforms.Resize(84,84).forward(t3)
let data = t3.data<float32>().ToArray()

let t5 = t2.reshape(resp.height,resp.width,1)
let t6 = t5.data<float32>().ToArray()
t6 = data

Array.zip resp.image_data_float (t2.data<float32>().ToArray())

c.Disconnect()





#r "nuget: MessagePack.FSharpExtensions"
#r "nuget: OpenCvSharp4.Windows, 4.6.0.20220608"
#r "nuget: TorchSharp-cpu"
#r "nuget: MathNet.Numerics.FSharp"
#load "../FsMsgPackRpc.fs"
#load "../AirSimCar.fs"

open System
let userProfile = Environment.GetEnvironmentVariable("UserProfile")
let nugetPath = @$"{userProfile}\.nuget"
let openCvLibPath = @$"{nugetPath}\packages\opencvsharp4.runtime.win\4.6.0.20220608\runtimes\win-64\native"
let path = Environment.GetEnvironmentVariable("path")
let path' = path + ";" + openCvLibPath
Environment.SetEnvironmentVariable("path",path')



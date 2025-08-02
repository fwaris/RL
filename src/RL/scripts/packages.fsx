﻿#r "nuget: MessagePack.FSharpExtensions"
#r "nuget: OpenCvSharp4.Windows, 4.6.0.20220608"
#r "nuget: TorchSharp-cuda-windows"
#r "nuget: TorchVision"
#r "nuget: MathNet.Numerics.FSharp"
#r "nuget: FSharpX.Collections"
#r "nuget: FSharp.Collections.ParallelSeq"
#r "nuget: TorchSharp.Fun"
#r "nuget: FsPickler"
#r "nuget: Plotly.NET"
#r "nuget: FSharp.Data"
#r "nuget: CAOpt"
#r "nuget: Microsoft.Diagnostics.NETCore.Client"
#r "nuget: Bezier"

#load "../SeqUtils.fs"
#load "../FsMsgPackRpc.fs"
//#load "../AirSimCar.fs"
//#load "../CarEnvironment.fs"
#load "../Experience.fs"
#load "../DQN.fs"

open System
let userProfile = Environment.GetEnvironmentVariable("UserProfile")
let nugetPath = @$"{userProfile}\.nuget"
let openCvLibPath = @$"{nugetPath}\packages\opencvsharp4.runtime.win\4.6.0.20220608\runtimes\win-64\native"
let path = Environment.GetEnvironmentVariable("path")
let path' = path + ";" + openCvLibPath
Environment.SetEnvironmentVariable("path",path')



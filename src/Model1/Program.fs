module Program
open TorchSharp
open System
open TorchSharp
open Types
open System.ComponentModel
open System
open FSharp.Control

//let mutable _ps = Unchecked.defaultof<_>

let startReRun parms =     
    async {
        try 
            let dTrain,dTest = Data.testTrain parms.TuneParms            
            let trainMarkets = Data.episodeLengthMarketSlices dTrain
            let plcy = Policy.policy parms
            let agent = Train.trainEpisodes parms plcy trainMarkets
            Test.evalModels parms
            System.GC.Collect()
        with ex -> 
            printfn "%A" (ex.Message,ex.StackTrace)
    }

let restartJobs() = 
        Model.parms.Value
        |> List.map (fun p -> 
            let o2 = p.DQN.Model.Online.Module.``to`` device.Value
            let tgt2 = p.DQN.Model.Target.Module.``to`` device.Value
            p
        )
        |> List.map startReRun
 
let MAX_PARALLEL = 2
let _run() =
    Test.clearModels()
    Data.resetLogs()
    let jobs = restartJobs() |> AsyncSeq.ofSeq
    jobs
    |> AsyncSeq.iterAsyncParallelThrottled MAX_PARALLEL id
    |> Async.Start

[<EntryPoint>]
let main args =
    verbosity <- LoggingLevel.Q
    printfn $"*** Server GC = {System.Runtime.GCSettings.IsServerGC}"
    TorchSharp.torch.InitializeDeviceType(TorchSharp.DeviceType.CUDA)
    let run() = async{ _run() } |> Async.Start
    let opt() =async{ Opt.optimize() } |> Async.Start
    printfn "o = optimize; r = run"
    let rec readCommandKey() =
        let k = System.Console.ReadKey()
        match k.KeyChar with 
        | 'o' -> opt(); readCommandKey()
        | 'r' -> run(); readCommandKey()
        | 'q' -> verbosity <- LoggingLevel.Q; readCommandKey()
        | 'l' -> verbosity <- LoggingLevel.L; readCommandKey()
        | 'h' -> verbosity <- LoggingLevel.H; readCommandKey()
        | 'm' -> verbosity <- LoggingLevel.M; readCommandKey()
        | _ when k.Key = ConsoleKey.Escape -> ()
        | x -> printfn $"'{x}' not recoqnized - q,l,h,m to set log level and esc to end"; readCommandKey()

    readCommandKey()
    0



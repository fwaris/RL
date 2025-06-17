module Program
open TorchSharp
open System
open TorchSharp
open Types
open System.ComponentModel
open System

let mutable _ps = Unchecked.defaultof<_>

let startReRun parms =     
    async {
        try 
            let dTrain,dTest = Data.testTrain parms.TuneParms            
            let trainMarkets = Data.episodeLengthMarketSlices dTrain
            let plcy = Policy.policy parms
            let agent = Train.trainEpisodes parms plcy trainMarkets
            _ps <- agent
            Test.evalModels parms
        with ex -> 
            printfn "%A" (ex.Message,ex.StackTrace)
    }

let restartJobs = 
    Model.parms     
    |> List.map (fun p -> 
        let o2 = p.DQN.Model.Online.Module.``to`` device        
        let tgt2 = p.DQN.Model.Target.Module.``to`` device
        p
    )
    |> List.map startReRun
 
let run() =
    Test.clearModels()
    Data.resetLogs()
    restartJobs |> Async.Parallel |> Async.Ignore |> Async.Start

verbosity <- LoggingLevel.L
printfn $"*** Server GC = {System.Runtime.GCSettings.IsServerGC}"
run()
//async{ Opt.optimize() } |> Async.Start

let rec readCommandKey() =
    let k = System.Console.ReadKey()
    match k.KeyChar with 
    | 'q' -> verbosity <- LoggingLevel.Q; readCommandKey()
    | 'l' -> verbosity <- LoggingLevel.L; readCommandKey()
    | 'h' -> verbosity <- LoggingLevel.H; readCommandKey()
    | 'm' -> verbosity <- LoggingLevel.M; readCommandKey()
    | _ when k.Key = ConsoleKey.Escape -> ()
    | x -> printfn $"'{x}' not recoqnized - q,l,h,m to set log level and esc to end"; readCommandKey()

readCommandKey()


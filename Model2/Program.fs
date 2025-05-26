module Program
open TorchSharp
open System
open TorchSharp
open Types

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
    restartJobs |> Async.Parallel |> Async.Ignore |> Async.RunSynchronously

verbosity <- LoggingLevel.L
printfn $"*** Server GC = {System.Runtime.GCSettings.IsServerGC}"
//run()
Opt.optimize()


module Program
open TorchSharp
open System
open TorchSharp
open Types

let NUM_MKT_SLICES tp = (Data.trainSize tp) / EPISODE_LENGTH

//printfn $"Running with {NUM_MKT_SLICES} market slices each of length {EPISODE_LENGTH} *  ; [left over {(Data.TRAIN_SIZE tp) % int NUM_MKT_SLICES}]"

let mutable _ps = Unchecked.defaultof<_>

let startReRun parms =     
    async {
        try 
            let data = Data.loadData parms.TuneParms
            let trainMarkets = Data.trainMarkets data
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
//run()
Opt.optimize()


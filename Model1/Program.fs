module Program
open TorchSharp
open System
open Types

let NUM_MKT_SLICES = Data.TRAIN_SIZE / EPISODE_LENGTH

printfn $"Running with {NUM_MKT_SLICES} market slices each of length {EPISODE_LENGTH} *  ; [left over {Data.TRAIN_SIZE % int NUM_MKT_SLICES}]"

let mutable _ps = Unchecked.defaultof<_>

let startReRun parms = 
    async {
        try 
            let plcy = Policy.policy parms
            let agent = Train.trainEpisodes parms plcy Data.trainMarkets
            _ps <- agent
        with ex -> 
            printfn "%A" (ex.Message,ex.StackTrace)
    }

let restartJobs = 
    Model.parms 
    |> List.map(fun p -> Policy.loadModel p |> Option.defaultValue p) 
    |> List.map (fun p -> 
        p.DQN.Model.Online.Module.``to`` torch_device |> ignore
        p.DQN.Model.Target.Module.``to`` torch_device |> ignore
        p)
    |> List.map startReRun
 
let run() =
    Test.clearModels()
    Data.resetLogs()
    restartJobs |> Async.Parallel |> Async.Ignore |> Async.RunSynchronously

verbosity <- LoggingLevel.L
run()
//Opt.optimize()


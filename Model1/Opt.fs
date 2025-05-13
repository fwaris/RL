module Opt
open System
open CA
open System.IO
open Types
open TorchSharp
open System.Text.RegularExpressions

let (|FileNumber|_|) (inp:string) = 
    let m = Regex.Match(inp,"opt_(\d+)\.csv")
    printfn $"{inp} - {if m.Success then m.NextMatch().Groups[1].Value else System.String.Empty}"
    if m.Success then m.Groups.[1].Value |> Some else None

let logFileName (folder:string) =
    Directory.GetFiles(folder,"opt*.csv")
    |> Array.choose(function FileNumber n -> Some n | _ -> None)
    |> Array.map int
    |> Array.sortDescending
    |> Array.tryHead
    |> Option.map (fun n -> $"opt_{n+1}.csv")
    |> Option.defaultValue $"opt.csv"

let OPT_LOG = lazy(root @@ (logFileName root))

//let clearLog () = if File.Exists OPT_LOG.Value then File.Delete OPT_LOG.Value

let toFloat basis low hi value =    
    (float hi - float low) / (float low) * basis * (float value)

let toVal = function
    | I(v,l,h) -> toFloat 0.1 l h v
    | F(v,_,_) -> v

let basis = 0.01
let toTParms (ps:float[]) =
    let trendWindowBars = (int ps.[8]) * 20
    {TuneParms.Default with
        GoodBuyInterReward = ps.[0]  * basis
        BadBuyInterPenalty = ps.[1] * basis
        ImpossibleBuyPenalty = ps.[2] * basis
        GoodSellInterReward = ps.[3] * basis
        BadSellInterPenalty = ps.[4] * basis
        ImpossibleSellPenalty = ps.[5] * basis
        NonInvestmentPenalty = ps.[6] * basis
        Layers = int64 ps.[7]
        TrendWindowBars  = trendWindowBars
        Lookback = int64 (trendWindowBars/3)
    }

let caparms = 
    [|                        
        I(80,60,100) //GoodBuyInterReward = 0.01
        I(-5,-30,-0) //BadBuyInterPenalty = -0.001
        I(-60,-100,-60) //ImpossibleBuyPenalty = -0.05
        I(80,60,100) //GoodSellInterReward = 0.01
        I(-50,-100,-0) //BadSellInterPenalty = - 0.001
        I(-60,-100,-60) //ImpossibleSellPenalty = -0.05
        I(-1,-10,0) //NonInvestmentPenalty = -0.0101                        
        I(5,1,5) //Layers
        I(3,1,3)  // TrendWindowBars 
    |]

let optLogger = MailboxProcessor.Start(fun inbox -> 
    async {
        while true do
            let! (gain:float, actDist:List<int*int>, tp:TuneParms) = inbox.Receive()
            let line = $"""{gain},"%A{actDist}",{tp.GoodBuyInterReward},{tp.BadBuyInterPenalty},{tp.ImpossibleBuyPenalty},{tp.GoodSellInterReward},{tp.BadSellInterPenalty},{tp.ImpossibleSellPenalty},{tp.NonInvestmentPenalty},{tp.Layers},{tp.TrendWindowBars},{tp.Lookback}"""
            try               
                if File.Exists OPT_LOG.Value |> not then
                    let header = $"""gain,actDist,GoodBuyInterReward,BadBuyInterPenalty,ImpossibleBuyPenalty,GoodSellInterReward,BadSellInterPenalty,ImpossibleSellPenalty,NonInvestmentPenalty,Layers,TendWindowBars,Lookback"""
                    File.AppendAllLines(OPT_LOG.Value,[header;line])
                else
                    File.AppendAllLines(OPT_LOG.Value,[line])
            with ex -> 
                printfn $"logger: {ex.Message}"
    })

let runOpt parms = 
    async {
        try 
            parms.DQN.Model.Online.Module.``to``(device) |> ignore
            parms.DQN.Model.Target.Module.``to``(device) |> ignore
            let plcy = Policy.policy parms
            let dTrain,dTest = Data.testTrain parms.TuneParms            
            let trainMarkets = Data.episodeLengthMarketSlices dTrain
            let testMarket = Data.singleMarketSlice dTest
            let agent = Train.trainEpisodes parms plcy trainMarkets
            let testGain,testDist = Test.evalModelTT parms.TuneParms parms.DQN.Model.Online testMarket
            printfn $"Gain; {testGain}; Test dist: {testDist}"
            optLogger.Post (testGain,testDist,parms.TuneParms)
            DQN.DQNModel.dispose parms.DQN.Model
            parms.Opt.Value.Dispose()
            let adjGain = 
                if testGain > 0.0 then testGain
                elif testGain = 0.0 then 
                    if testDist.Length > 1 then testGain + (float testDist.Length * 0.001) else testGain
                else 
                    testGain
            return adjGain
        with ex -> 
            printfn "%A" (ex.Message,ex.StackTrace)
            return -0.9999
    }

let mutable _id = 0
let nextId () = System.Threading.Interlocked.Increment &_id

let fopt (parms:float[]) =
    async {
        let tp = toTParms parms
        let baseParms = Model.parms1 (nextId()) (0.001, tp )                   //every fitness evaluation needs separate optimizer and model
        let baseParms = {baseParms with LogSteps=false; SaveModels=false}
        let optParms = {baseParms with TuneParms = tp}
        let! gain = runOpt optParms
        System.GC.Collect()
        return gain
    }

let appendStepNumber (step:TimeStep<_>) = 
    let fn = root @@ "steps.Text"
    let line = $"{DateTime.Now}{step.Count},{step.Best.Head.MFitness},{step.Best.Head.MParms}"
    File.AppendAllText(fn,line)

let optimize() =
    //clearLog() //add new files for each run
    let fitness ps = fopt ps |> Async.RunSynchronously    
    let mutable step = CALib.API.initCA(caparms, fitness , Maximize, popSize=36, beliefSpace = CALib.BeliefSpace.Hybrid)
    for i in 0 .. 15000 do         
        printfn $"
************************************************
CA STEP {i}
************************************************"
        //step <- CALib.API.Step step
        step <- CALib.API.Step(step, maxParallelism=5)
        appendStepNumber step


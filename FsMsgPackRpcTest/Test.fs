module Test
open System
open System.IO
open Plotly.NET
open TorchSharp
open TorchSharp.Fun
open DQN
open RL
open Types

let interimModel = root @@ "test_DQN.bin"

let saveInterim parms =    
    DQN.DQNModel.save interimModel parms.DQN.Model

let testMarket() = {prices = Data.dataTest}
let trainMarket() = {prices = Data.dataTrain}

let evalModelTT (model:IModel) (market:MarketSlice) = 
    let s = AgentState.Default -1 0.0 1_000_000 
    let exp = Exploration.Default        
    let dataChunks = market.Market.prices |> Array.windowed (int LOOKBACK)
    let modelDevice = model.Module.parameters() |> Seq.head |> fun t -> t.device        
    let s',actionsTaken = 
        ((s,[]),dataChunks) 
        ||> Array.fold (fun (s,acc) bars -> 
            let inp = bars |> Array.collect (fun b -> [|b.TrendLong;b.TrendShort;b.NOpen;b.NHigh;b.NLow;b.NClose|])    
            use d_inp = torch.tensor(inp,dimensions=ReadOnlySpan [|1L;LOOKBACK;INPUT_DIM|], dtype=torch.float32)
            //let t_d_inp = Tensor.getDataNested<float32> d_inp
            use d_inp = d_inp.``to``(modelDevice)
            use q = model.forward d_inp
            //let t_q = Tensor.getData<float32> q
            let act = q.argmax(-1L).flatten().ToInt32()               
            let s = 
                match act with              //just execute buy / sell actions
                | 0 -> Agent.buy market s
                | 1 -> Agent.sell market s
                | _ -> s
            {s with TimeStep = s.TimeStep + 1},act::acc)
    let lastPrice = market.LastBar.Bar |> Data.avgPrice
    let sGain = ((lastPrice * float s'.Stock + s'.CashOnHand) - s'.InitialCash) / s'.InitialCash
    let years = (market.LastBar.Bar.Time - (market.Market.prices.[0].Bar.Time)).TotalDays / 365.0
    let annualizedGain = sGain / years
    annualizedGain,actionsTaken
    //printfn $"model: {modelFile}, gain: {gain}, adjGain: {adjGain}"
    //modelFile,adjGain

let evalModel parms (name:string) (model:IModel) =
    try
        model.Module.eval()
        let testMarket = let tm = testMarket() in {Market=tm; StartIndex=0; EndIndex=tm.prices.Length-1}
        let trainMarket = let tm = trainMarket() in {Market=tm; StartIndex=0; EndIndex=tm.prices.Length-1}            
        let gainTest,actTest = evalModelTT model testMarket 
        let gainTrain,actTrain = evalModelTT model trainMarket
        let testDist = actTest |> List.countBy id |> List.sortBy fst
        let trainDist = actTrain |> List.countBy id |> List.sortBy fst
        printfn $"Emodel: {parms.RunId} {name}, Annual Gain -  Test: %0.3f{gainTest}, Train: $0.3f{gainTrain}"
        printfn $"Test dist: {testDist}; Train dist: {trainDist}"
        name,gainTest,gainTrain,testDist
    finally
        model.Module.train()
    
let evalModelFile parms modelFile  =
    let model = (DQN.DQNModel.load parms.CreateModel modelFile).Online
    evalModel parms modelFile model

let copyModels() =
    let dir = root @@ "models_eval" 
    if Directory.Exists dir |> not then Directory.CreateDirectory dir |> ignore
    dir |> Directory.GetFiles |> Seq.iter File.Delete        
    let dirIn = Path.Combine(root,"models")
    Directory.GetFiles(dirIn,"*.bin")
    |> Seq.map FileInfo
    |> Seq.sortByDescending (fun f->f.CreationTime)
    |> Seq.truncate 1                                 //most recent 50 models
    |> Seq.map (fun f->f.FullName)
    |> Seq.iter (fun f->File.Copy(f,Path.Combine(dir,Path.GetFileName(f)),true))

let evalModels parms = 
    copyModels()
    let evals = 
        Directory.GetFiles(Path.Combine(root,"models_eval"),"*.bin")
        |> Seq.map (evalModelFile parms)
        |> Seq.toArray
    evals
    |> Seq.iter (fun (n,gainTst,gainTrain,dist) -> 
        let dist = dist |> List.map (fun (a,b) -> string a, b)
        dist |> Chart.Column |> Chart.withTitle $"Test action dist {n}, gain: %0.3f{gainTst}" |> Chart.show
    )
    (*
    evals
    |> Seq.map (fun (m,tst,trn,_) -> tst)
    |> Chart.Histogram
    |> Chart.show
    evals
    |> Seq.map (fun (m,tst,trn,_) -> trn,tst)
    |> Chart.Point
    |> Chart.withXAxisStyle "Train"
    |> Chart.withYAxisStyle "Test"
    |> Chart.show
    *)

let runTest parms = 
    saveInterim parms
    evalModel parms interimModel

let clearModels() = 
    let mdir = root @@ "models"
    let edir = root @@ "models_eval"
    if Directory.Exists mdir then mdir |> Directory.GetFiles |> Seq.iter File.Delete
    if Directory.Exists edir then edir |> Directory.GetFiles |> Seq.iter File.Delete

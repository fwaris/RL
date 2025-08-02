﻿module Test
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

let runAgent (policy:IModel) (market:MarketSlice) (s:AgentState) = 
    let device = policy.Module.parameters() |> Seq.tryHead |> Option.map _.device |> Option.defaultValue torch.CPU
    let s' = Agent.getObservations () market s
    use state = s'.CurrentState.unsqueeze(0)
    use state' = state.``to``(device)
    use actionVals  = policy.forward(state')
    let action = actionVals.argmax(1L).ToInt32()
    let s'' = Agent.doAction () market s' action
    s'',action

let evalModelTT tp (policy:IModel) (market:MarketSlice) =
    let sInit = AgentState.Default -1 0.0 INITIAL_CASH tp
    let rec loop actions s = 
        if market.IsDone (s.TimeStep + 1) then 
            let lastBar = market.LastBar
            let avgPrice = Data.avgPrice lastBar.Bar
            let equity = s.CashOnHand + (avgPrice * s.Stock)
            let gain = (equity - s.InitialCash) / s.InitialCash
            let years = (market.LastBar.Bar.Time - (market.Market.prices.[0].Bar.Time)).TotalDays / 365.0
            let annualizedGain = gain / years
            let actDist = actions |> List.countBy id |> List.sortBy fst
            actDist,annualizedGain,s
        else
            let s'',action = runAgent policy market s
            loop (action::actions) s''
    let actions,gain,lastAgentState = loop [] sInit
    gain,actions,lastAgentState

let evalModel parms (name:string) (model:IModel) =
    try
        model.Module.eval()
        let dTrain,dTest = Data.testTrain parms.TuneParms
        let testMarket =  Data.singleMarketSlice dTrain
        let trainMarket = Data.singleMarketSlice dTest
        let gainTest,testDist,testAgentState = evalModelTT parms.TuneParms model testMarket 
        let gainTrain,trainDist,_ = evalModelTT parms.TuneParms model trainMarket
        printfn $"Emodel: {parms.RunId} {name}, Annual Gain -  Test: %0.3f{gainTest}, Train: %0.3f{gainTrain}"
        printfn $"Test dist: {testDist}; Train dist: {trainDist}"
        name,gainTest,gainTrain,testDist,testAgentState
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
    |> Seq.iter (fun (n,gainTst,gainTrain,dist,agentState) -> 
        let dist = dist |> List.map (fun (a,b) -> string a, b)
        dist |> Chart.Column |> Chart.withTitle $"Test action dist {n}, gain: %0.3f{gainTst}" |> Chart.show
        let actions = agentState.AgentBook |> List.rev |> List.choose (fun x -> x.NBar |> Option.map (fun b-> b,x.Action))
        [
            actions |> List.map (fun (b,a) -> b.Bar.Time,Data.avgPrice b.Bar) |> Chart.Line
            actions 
            |> List.filter (fun (b,a) -> a=0) 
            |> List.map (fun (b,a) -> b.Bar.Time, Data.avgPrice b.Bar) 
            |> Chart.Point 
            |> Chart.withMarkerStyle (Symbol=StyleParam.MarkerSymbol.ArrowBarUp)
            actions 
            |> List.filter (fun (b,a) -> a=1) 
            |> List.map (fun (b,a) -> b.Bar.Time, Data.avgPrice b.Bar) 
            |> Chart.Point 
            |> Chart.withMarkerStyle (Symbol=StyleParam.MarkerSymbol.ArrowBarDown)
        ]
        |> Chart.combine
        |> Chart.withSize(1000.,700)
        |> Chart.show
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

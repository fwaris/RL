module Test
open System
open System.IO
open Plotly.NET
open TorchSharp
open TorchSharp.Fun
open DQN
open RL
open Types
open System.Text.RegularExpressions

let interimModel = root @@ "test_DQN.bin"

//let saveInterim parms =    
//    DQN.DQNModel.save interimModel parms.DQN.Model

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
            let avgPrice = Data.effectivePrice lastBar.Bar
            let equity = s.CashOnHand + (avgPrice * s.Stock)
            let gain = (equity - s.InitialCash) / s.InitialCash
            let years = (market.LastBar.Bar.Time - (market.Market.prices.[0].Bar.Time)).TotalDays / 365.0
            let annualizedGain = gain / years
            let actDist = actions |> List.countBy id |> List.sortBy fst
            actDist,annualizedGain
        else
            let s'',action = runAgent policy market s
            loop (action::actions) s''
    let actions,gain = loop [] sInit
    gain,actions

let evalModel parms (name:string) (model:IModel) =
    try
        model.Module.eval()
        let dTrain,dTest = Data.testTrain parms.TuneParms
        let testMarket =  Data.singleMarketSlice dTrain
        let trainMarket = Data.singleMarketSlice dTest
        let gainTest,testDist = evalModelTT parms.TuneParms model testMarket 
        let gainTrain,trainDist = evalModelTT parms.TuneParms model trainMarket
        printfn $"Emodel: {parms.RunId} {name}, Annual Gain -  Test: %0.3f{gainTest}, Train: %0.3f{gainTrain}"
        printfn $"Test dist: {testDist}; Train dist: {trainDist}"
        name,gainTest,gainTrain,testDist
    finally
        model.Module.train()
    
let extractModelInfo (path: string) =
    let pattern = @"model_(\w+)_(\d+)_(\d+)\.bin$"
    let m = Regex.Match(path, pattern)
    if m.Success then
        let runId = string m.Groups.[1].Value
        let epoch = int m.Groups.[2].Value
        let stepNum = int m.Groups.[3].Value
        Some (runId, epoch, stepNum)
    else
        None

let getLatestModel parms dir = 
    Directory.GetFiles(dir,"*.bin") 
    |> Seq.toList
    |> List.choose (fun path -> extractModelInfo path |> Option.map(fun (id,epoch,step) -> id,(step,path)))
    |> List.filter (fun (id,_) -> id = parms.RunId)
    |> List.sortByDescending (snd>>fst)
    |> List.map (snd>>snd)
    |> List.tryHead

let evalModelFile parms =
    let dir = root @@ "models_eval"
    match getLatestModel parms dir with 
    | Some modelFile -> 
        let model = (DQN.DQNModel.load parms.CreateModel modelFile).Online        
        let n,gainTst,gainTrain,dist = evalModel parms modelFile model
        dist |> Chart.Column |> Chart.withTitle $"Id {parms.RunId} Test<br>Gain test: %0.3f{gainTst}" |> Chart.show
    | None -> printfn $"no model file found for parms id {parms.RunId}"

let private deleteExistingEval parms dir =
    Directory.GetFiles(dir,"*.bin")
    |> Seq.choose (fun path -> extractModelInfo path |> Option.map (fun (id,_,_) -> id,path))
    |> Seq.toList
    |> List.filter (fun (id,path) -> parms.RunId = id)
    |> List.iter (fun (_,path) -> File.Delete path)

let private copyLatestModelToEval parms modelDir evalDir = 
    getLatestModel parms modelDir
    |> Option.iter (fun f->File.Copy(f,Path.Combine(evalDir,Path.GetFileName(f)),true))

let copyModels (parms:Parms) =
    let evalDir = root @@ "models_eval" 
    let modelDir = root @@ "models"
    if Directory.Exists evalDir |> not then Directory.CreateDirectory evalDir |> ignore
    deleteExistingEval parms evalDir    
    copyLatestModelToEval parms modelDir evalDir

let evalModels parms = 
    copyModels parms
    evalModelFile parms

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

//let runTest parms = 
//    saveInterim parms
//    evalModel parms interimModel

let clearModels() = 
    let mdir = root @@ "models"
    let edir = root @@ "models_eval"
    if Directory.Exists mdir then mdir |> Directory.GetFiles |> Seq.iter File.Delete
    if Directory.Exists edir then edir |> Directory.GetFiles |> Seq.iter File.Delete

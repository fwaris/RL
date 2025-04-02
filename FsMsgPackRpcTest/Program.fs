//#load "../scripts/packages.fsx"
//#load "../TsData.fs"
//#load "../RL.fs"
//open System.Threading.Tasks
open MathNet.Numerics
open TorchSharp
open TorchSharp.Fun
open TsData
open FSharpx.Collections
open RL
open System.IO
open Plotly.NET
open DQN
open System
open FSharp.Collections.ParallelSeq
open SeqUtils
let ( @@ ) a b = Path.Combine(a,b)

let TREND_WINDOW = 20L
let REWARD_HORIZON_BARS = 10
let LOOKBACK = 30L
let TX_COST_CNTRCT = 0.5
let MAX_TRADE_SIZE = 25.
let EPISODE_LENGTH = 336 //*4
let INPUT_DIM = 6L
let TRAIN_FRAC = 0.7
let ACTIONS = 3 //0,1,2 - buy, sell, hold
let device = if torch.cuda_is_available() then torch.CUDA else torch.CPU
let data_dir = System.Environment.GetEnvironmentVariable("DATA_DRIVE")
let root = data_dir @@ @"s\tradestation"
let INPUT_FILE = root @@ "mes_hist_td2.csv"

type LoggingLevel = Q | L | M | H 
    with  
        member this.IsLow = match this with L | M | H -> true | _ -> false
        member this.isHigh = match this with H -> true | _ -> false
        member this.IsMed = match this with M | H -> true | _ -> false

let mutable verbosity = LoggingLevel.Q

type Parms =
    {
        LearnRate        : float
        CreateModel      : unit -> IModel                   //need model creation function so that we can load weights from file
        DQN              : DQN
        LossFn           : Loss<torch.Tensor,torch.Tensor,torch.Tensor>
        Opt              : torch.optim.Optimizer
        LearnEverySteps  : int
        SyncEverySteps   : int
        BatchSize        : int
        Epochs           : int
        RunId            : int
    }
    with 
        static member Default modelFn ddqn lr id = 
            let mps = ddqn.Model.Online.Module.parameters()
            {
                LearnRate       = lr
                CreateModel     = modelFn
                DQN             = ddqn
                LossFn          = torch.nn.SmoothL1Loss()
                Opt             = torch.optim.Adam(mps, lr=lr,weight_decay=0.00001)
                LearnEverySteps = 3
                SyncEverySteps  = 1000
                BatchSize       = 32
                Epochs          = 6
                RunId           = id
                
            }

//keep track of the information we need to run RL in here
type RLState =
    {
        AgentId          : int
        TimeStep         : int
        State            : torch.Tensor
        PrevState        : torch.Tensor
        Step             : Step
        InitialCash      : float
        Stock            : float
        TradeSize        : float
        CashOnHand       : float
        LookBack         : int64
        ExpBuff          : DQN.ExperienceBuffer
        S_reward         : float
        S_gain           : float
        Episode          : int
        AvgLoss          : float
        Actions          : ActionResult list
        IsDone           : bool
    }
    with 
        ///reset for new episode
        static member Reset x = 
            let a = 
                {x with 
                    //Step            = {x.Step with Num=0} //keep current exploration rate; just update step number
                    TimeStep        = 0
                    CashOnHand      = x.InitialCash
                    Stock           = 0
                    State           = torch.zeros([|x.LookBack;INPUT_DIM|],dtype=Nullable torch.float32)
                    PrevState       = torch.zeros([|x.LookBack;INPUT_DIM|],dtype=Nullable torch.float32)
                    Actions         = []
                }            
            if verbosity.IsLow then 
                printfn  $"Reset called {x.AgentId} x={x.Step.ExplorationRate} a={a.Step.ExplorationRate}"
            a

        static member Default agentId initExpRate initialCash = 
            let expBuff = {DQN.Buffer=RandomAccessList.empty; DQN.Max=50000}
            {
                TimeStep         = 0
                AgentId          = agentId
                State            = torch.zeros([|LOOKBACK;INPUT_DIM|],dtype=Nullable torch.float32)
                PrevState        = torch.zeros([|LOOKBACK;INPUT_DIM|],dtype=Nullable torch.float32)
                Step             = {ExplorationRate = initExpRate; Num=0}
                Stock            = 0
                TradeSize        = 0.0
                CashOnHand       = initialCash
                InitialCash      = initialCash
                LookBack         = LOOKBACK
                ExpBuff          = expBuff
                S_reward         = -1.0
                S_gain           = -1.0
                Episode          = 0
                AvgLoss          = 0.0
                IsDone           = false
                Actions          = []
            }

type NBar =
    {
        TrendShort : float
        TrendLong : float
        NOpen  : float 
        NHigh  : float
        NLow   : float
        NClose : float
        NVolume : float
        Bar  : Bar
    }
//environment

type Prices = {prices : NBar array}
    with 
        member this.IsDone t = t >= this.prices.Length 
        member this.Bar t = if t < this.prices.Length then Some this.prices.[t] else None

type MarketSlice = {Market:Prices; StartIndex:int; EndIndex:int}
    with 
        member this.IsDone t = t + this.StartIndex >= this.EndIndex
        member this.Bar t = this.Market.Bar (t + this.StartIndex)
        member this.LastBar = this.Market.prices.[this.EndIndex]
        member this.Length = this.EndIndex - this.StartIndex + 1

type StepResult = {Market:MarketSlice; Rl:RLState; ActionResult:ActionResult}


module Data = 
    let avgPrice bar = 0.5 * (bar.High + bar.Low)        

    let isNaN (c:float) = Double.IsNaN c || Double.IsInfinity c

    let clipSlope (x:float) = 
        tanh (x/5.0)
        //max -5.0 (min 5.0 x) //clip slope to [-5,5]

    let loadData() = 
        let data =
            File.ReadLines INPUT_FILE
            |> Seq.filter (fun l -> String.IsNullOrWhiteSpace l |> not)
            |> Seq.map(fun l -> 
                let xs = l.Split(',')
                let d =
                    {
                        Time = DateTime.Parse xs.[1]
                        Open = float xs.[2]
                        High = float xs.[3]
                        Low = float xs.[4]
                        Close = float xs.[5]
                        Volume = float xs.[6]
                    }
                d)
            |> Seq.toList
        let pd = data |> List.windowed 200 //|> List.truncate (100000 * 4)
        let pds =
            pd
            |> List.mapi (fun i xs ->
                let x = List.last xs
                let y = xs.[xs.Length - 2]
                let pts1 = xs |> List.map avgPrice
                let ys1N = LinearAlgebra.Double.Vector.Build.DenseOfEnumerable(pts1).Normalize(1.0)
                let xs1N = LinearAlgebra.Double.Vector.Build.DenseOfEnumerable([0 .. pts1.Length]).Normalize(1.0)
                let pts2 = xs |> List.skip (xs.Length/2) |> List.map avgPrice
                let ys2N = LinearAlgebra.Double.Vector.Build.DenseOfEnumerable(pts2).Normalize(1.0)
                let xs2N = LinearAlgebra.Double.Vector.Build.DenseOfEnumerable([0 .. pts2.Length]).Normalize(1.0)
                let struct(_,s1) = LinearRegression.SimpleRegression.Fit(Seq.zip xs1N ys1N) 
                let struct(_,s2) = LinearRegression.SimpleRegression.Fit(Seq.zip xs2N ys2N)
                let cs1 = clipSlope s1
                let cs2 = clipSlope s2
                let d =
                    {
                        TrendLong = cs1
                        TrendShort = cs2
                        NOpen = exp(y.Open  / x.Open) 
                        NHigh = exp(y.High /  x.High) 
                        NLow =  exp(y.Low / x.Low)   
                        NClose = exp(y.Close / x.Close)
                        NVolume = exp(y.Volume / x.Volume) 

                        //TrendLong = cs1
                        //TrendShort = cs2
                        //NOpen = log(y.Open/x.Open) |> max -18. //- 1.0
                        //NHigh = log(y.High/x.High) |> max -18. //- 1.0
                        //NLow =  log(y.Low/x.Low)   |> max -18. //- 1.0
                        //NClose = log(y.Close/x.Close) |> max -18.// - 1.0
                        //NVolume = log(y.Volume/x.Volume) |> max -18. //- 1.0
                        //NOpen = (y.Open/x.Open) //|> max -18. //- 1.0
                        //NHigh = (y.High/x.High) //|> max -18. //- 1.0
                        //NLow =  (y.Low/x.Low)   //|> max -18. //- 1.0
                        //NClose = (y.Close/x.Close) //|> max -18.// - 1.0
                        //NVolume = (y.Volume/x.Volume) //|> max -18. //- 1.0
                        Bar  = y
                    }
                if isNaN d.NOpen ||isNaN d.NHigh || isNaN d.NLow || isNaN d.NClose || isNaN d.NVolume then
                    failwith "nan in data"
                (x,y),d
            )
        let xl = pds |> List.last
        pds |> List.map snd

    let dataRaw = loadData() |> List.truncate (EPISODE_LENGTH * 10)
    let TRAIN_SIZE = float dataRaw.Length * 0.7 |> int
    let dataTrain = dataRaw |> Seq.truncate TRAIN_SIZE |> Seq.toArray
    let dataTest = dataRaw |> Seq.skip TRAIN_SIZE |> Seq.toArray
    let pricesTrain = {prices = dataTrain}
    let pricesTest = {prices = dataTest}
    
    let resetLogs() =
        let logDir = root @@ "logs"
        if Directory.Exists logDir |> not then 
            Directory.CreateDirectory logDir |> ignore
        else
            Directory.GetFiles(logDir) |> Seq.iter File.Delete

    let logger = MailboxProcessor.Start(fun inbox -> 
        async {
            while true do
                let! (agentId:int,parmsId:int,line:string) = inbox.Receive()
                try
                    let fn = root @@ "logs" @@ $"log_{agentId}_{parmsId}.csv"
                    if File.Exists fn |> not then
                        //let logLine = $"{s.AgentId},{s.Episode},{s.Step.Num},{action},{avgP},{s.CashOnHand},{s.Stock},{reward},{sGain},{parms.RunId}"
                        let header = "agentId,episode,step,action,price,cash,stock,reward,gain,parmId"
                        File.AppendAllLines(fn,[header;line])
                    else
                        File.AppendAllLines(fn,[line])
                with ex -> 
                    printfn $"logger: {ex.Message}"
        })



let NUM_AGENTS = Data.TRAIN_SIZE / EPISODE_LENGTH
//let NUM_AGENTS = 1

printfn $"Episode size {EPISODE_LENGTH} * agents {NUM_AGENTS}; [left over {Data.TRAIN_SIZE % int NUM_AGENTS}]"

//Properties not expected to change over the course of the run (e.g. model, hyperparameters, ...)
//can support multiple concurrent runs

module Agent = 
    open DQN
    let bar (env:MarketSlice) t = env.Bar t
    

    let buy (env:MarketSlice) (s:RLState) = 
        bar env s.TimeStep
        |> Option.map (fun nbar -> 
            let avgP = Data.avgPrice nbar.Bar
            let priceWithCost = avgP + TX_COST_CNTRCT
            let stockToBuy = s.CashOnHand / priceWithCost |> floor |> max 0. |> min MAX_TRADE_SIZE
            let outlay = stockToBuy * priceWithCost
            let coh = s.CashOnHand - outlay |> max 0.            
            let stock = s.Stock + stockToBuy 
            assert (stock >= 0.)
            {s with CashOnHand=coh; Stock=stock; TradeSize = stockToBuy})
        |> Option.defaultValue s

    let sell (env:MarketSlice) (s:RLState) =
        bar env s.TimeStep
        |> Option.map (fun nbar -> 
            let avgP = Data.avgPrice nbar.Bar
            let priceWithCost = avgP - TX_COST_CNTRCT
            let stockToSell = s.Stock |> min MAX_TRADE_SIZE
            let inlay = stockToSell * priceWithCost
            let coh = s.CashOnHand + inlay
            let remStock = s.Stock - stockToSell |> max 0.
            {s with CashOnHand=coh; Stock=remStock; TradeSize = -stockToSell})
        |> Option.defaultValue s

    let doAction _ env s act =
        let s = 
            match act with
            | 0 -> buy env s
            | 1 -> sell env s
            | _ -> s                //hold
        {s with TimeStep = s.TimeStep + 1}

    let skipHead = torch.TensorIndex.Slice(1)

    let getObservations _ (env:MarketSlice) (s:RLState) =         
        let b =  bar env s.TimeStep |> Option.defaultWith (fun () -> failwith "bar not found")
        let t1 = torch.tensor([|b.TrendLong;b.TrendShort;b.NOpen;b.NHigh;b.NLow;b.NClose|],dtype=torch.float32)
        let ts = torch.vstack([|s.State;t1|])
        let ts2 = if ts.shape.[0] > s.LookBack then ts.index skipHead else ts  // 40 x 5             
        {s with State = ts2; PrevState = s.State}
        
    let computeRewards parms env s action =         
        bar env s.TimeStep
        |> Option.map (fun cBar -> 
            let avgP     = Data.avgPrice  cBar.Bar
            let futurePrices = [s.TimeStep+1 .. s.TimeStep + REWARD_HORIZON_BARS] |> List.choose (bar env) |> List.map _.Bar |> List.map Data.avgPrice
            let intermediateReward = 
                if action = 0 && futurePrices |> List.exists (fun p -> p >= avgP + TX_COST_CNTRCT) then
                    0.0005
                elif action = 1 && futurePrices |> List.exists (fun p -> p <= avgP - TX_COST_CNTRCT) then
                    0.0005
                else
                    -0.0001
            let sGain    = ((avgP * float s.Stock + s.CashOnHand) - s.InitialCash) / s.InitialCash
            let isDone   = env.IsDone (s.TimeStep + 1)
            let reward  = 
                if isDone then 
                    sGain 
                else 
                    intermediateReward         
            if verbosity.isHigh then
                printfn $"{s.AgentId}-{s.TimeStep}|{s.Step.Num} - P:%0.3f{avgP}, OnHand:{s.CashOnHand}, S:{s.Stock}, R:{reward}, A:{action}, Exp:{s.Step.ExplorationRate} Gain:{sGain}"
            let logLine = $"{s.AgentId},{s.Episode},{s.TimeStep},{action},{avgP},{s.CashOnHand},{s.Stock},{reward},{sGain},{parms.RunId}"
            Data.logger.Post (s.AgentId,parms.RunId,logLine)
            let experience = {NextState = s.State; Action=action; State = s.PrevState; Reward=float32 reward; Done=isDone }
            let experienceBuff = Experience.append experience s.ExpBuff  
            {s with ExpBuff = experienceBuff; S_reward=reward; S_gain = sGain; IsDone=isDone},isDone,reward
        )
        |> Option.defaultValue (s,false,0.0)
       

    let agent  = 
        {
            doAction = doAction
            getObservations = getObservations
            computeRewards = computeRewards
        }

module Policy =

    let updateQ parms (losses:torch.Tensor []) =        
        parms.Opt.zero_grad()
        let losseD = losses |> Array.map (fun l -> l.backward(); l.ToDouble())
        use t = parms.Opt.step() 
        let avgLoss = losseD |> Array.average
        if Double.IsNaN avgLoss then
            let pns = parms.DQN.Model.Online.Module.named_parameters() |> Seq.map(fun struct(n,x) -> n, Tensor.getDataNested<float32> x) |> Seq.toArray
            ()
            failwith "Nan loss"
        avgLoss

    let loss parms s = 
        let states,nextStates,rewards,actions,dones = Experience.recall parms.BatchSize s.ExpBuff  //sample from experience
        use states = states.``to``(parms.DQN.Device)
        use nextStates = nextStates.``to``(parms.DQN.Device)
        let td_est = DQN.td_estimate states actions parms.DQN           //estimate the Q-value of state-action pairs from online model
        let td_tgt = DQN.td_target rewards nextStates dones parms.DQN   //
        let loss = parms.LossFn.forward(td_est,td_tgt)
        if loss.ToDouble() |> Double.IsNaN then 
            let t_states = Tensor.getDataNested<float32> states
            let t_nextStates = Tensor.getDataNested<float32> nextStates
            let t_states = Tensor.getDataNested<float32> states
            let t_td_est = Tensor.getDataNested<float32> td_est
            let t_td_tgt = Tensor.getDataNested<float32> td_tgt
            ()
        loss
    let ensureDirForFilePath (file:string) = 
        let dir = Path.GetDirectoryName(file)
        if dir |> Directory.Exists |> not then Directory.CreateDirectory(dir) |> ignore

    let ensureDir (dir:string) = 
       if Directory.Exists dir |> not then Directory.CreateDirectory(dir) |> ignore

    let loadModel parms (device:torch.Device) =
        let dir = root @@ "models_restart"
        ensureDir dir
        Directory.GetFiles(dir,$"model_{parms.RunId}*") |> Seq.sortByDescending (fun f -> (FileInfo f).LastWriteTime) |> Seq.tryHead
        |> Option.map(fun fn ->
            let mdl  = DQN.DQNModel.load parms.CreateModel fn                        
            mdl.Online.Module.``to``(device) |> ignore
            mdl.Target.Module.``to``(device) |> ignore
            let dqn = {parms.DQN with Model = mdl; }
            {parms with DQN = dqn})


    let syncModel parms s = 
        DQNModel.sync parms.DQN.Model parms.DQN.Device
        let fn = root @@ "models" @@ $"model_{parms.RunId}_{s.Episode}_{s.Step.Num}.bin"
        ensureDirForFilePath fn
        DQNModel.save fn parms.DQN.Model 
        if verbosity.IsLow then printfn "Synced"

    let rec policy parms = 
        {
            selectAction = fun parms (s:RLState) -> 
                let act =  DQN.selectAction s.State parms.DQN s.Step
                act

            update = fun parms st_act_done_rwd  ->      
                let losses = st_act_done_rwd |> Seq.map (fun (s,_) -> loss parms s) |> Seq.toArray
                let avgLoss = updateQ parms losses
                if Double.IsNaN avgLoss then
                    let ls1 = losses |> Array.map(Tensor.getData<float32>)
                    ()
                if verbosity.IsMed then printfn $"avg loss {avgLoss}"
                let rs = st_act_done_rwd |> List.map fst |> List.map(fun r -> {r with AvgLoss = avgLoss})
                policy parms, rs

            sync = syncModel
        }
        
module Test = 
    let interimModel = root @@ "test_DQN.bin"

    let saveInterim parms =    
        DQN.DQNModel.save interimModel parms.DQN.Model

    let testMarket() = {prices = Data.dataTest}
    let trainMarket() = {prices = Data.dataTrain}

    let evalModelTT (model:IModel) (market:MarketSlice) = 
        let s = RLState.Default -1 0.0 1_000_000 
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
            printfn $"model: {parms.RunId} {name}, Annual Gain -  Test: {gainTest}, Train: {gainTrain}"
            printfn $"Test dist: {testDist}; Train dist: {trainDist}"
            name,gainTest,gainTrain
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
        |> Seq.truncate 50                                  //most recent 50 models
        |> Seq.map (fun f->f.FullName)
        |> Seq.iter (fun f->File.Copy(f,Path.Combine(dir,Path.GetFileName(f)),true))

    let evalModels parms = 
        copyModels()
        let evals = 
            Directory.GetFiles(Path.Combine(root,"models_eval"),"*.bin")
            |> Seq.map (evalModelFile parms)
            |> Seq.toArray
        evals
        |> Seq.map (fun (m,tst,trn) -> tst)
        |> Chart.Histogram
        |> Chart.show
        evals
        |> Seq.map (fun (m,tst,trn) -> trn,tst)
        |> Chart.Point
        |> Chart.withXAxisStyle "Train"
        |> Chart.withYAxisStyle "Test"
        |> Chart.show

    let runTest parms = 
        saveInterim parms
        evalModel parms interimModel

    let clearModels() = 
        let mdir = root @@ "models"
        let edir = root @@ "models_eval"
        if Directory.Exists mdir then mdir |> Directory.GetFiles |> Seq.iter File.Delete
        if Directory.Exists edir then edir |> Directory.GetFiles |> Seq.iter File.Delete

let acctBlown (s:RLState) = s.CashOnHand < 10000.0 && s.Stock <= 0
let isDone (m:MarketSlice,s) = m.IsDone (s.TimeStep+1) || acctBlown s

//single step a single agent in its associated market
let stepAgent parms plcy (m,s) = 
    let s',ar = step parms m Agent.agent plcy s                           
    let s'' = {s' with Step = DQN.updateStep parms.DQN.Exploration s'.Step; Actions=ar::s.Actions} // update step number and exploration rate for each agent
    {Market=m; Rl=s''; ActionResult=ar}

let updatePolicy parms plcy rs =
    let r0 = rs |> List.find (fun x -> not x.Rl.IsDone)
    if r0.Rl.Step.Num % parms.LearnEverySteps = 0 then 
        let remainAgents = rs |> List.filter(fun r -> r.Rl.IsDone |> not) |> List.map (fun m -> m.Rl,m.ActionResult) //update policy for agents still in play
        let _,updateAgents = plcy.update parms remainAgents
        let updateAgentsMap = updateAgents |> List.map(fun r -> r.AgentId,r) |> Map.ofList
        rs |> List.map(fun x -> updateAgentsMap |> Map.tryFind x.Rl.AgentId |> Option.map(fun r -> {x with Rl=r}) |> Option.defaultValue x) //merge update with others
    else
        rs

let syncModel parms plcy runStates = 
    runStates
    |> List.tryFind(fun x -> x.Rl.IsDone)
    |> Option.bind (fun r -> if r.Rl.Step.Num > 0 && r.Rl.Step.Num % parms.SyncEverySteps = 0 then Some r.Rl else None)
    |> Option.iter(fun r  -> plcy.sync parms r)
            
let runEpisodes  parms plcy (ms:(MarketSlice*RLState) list) =
    let rec loop ms =
        let ms' = ms |> List.map (stepAgent parms plcy) // to debug run serially
        let allDone = ms' |> List.forall (fun m -> m.Rl.IsDone)
        System.GC.Collect()
        if not allDone then
            let rs = updatePolicy parms plcy ms'
            syncModel parms plcy rs
            let ms = rs |> List.map(fun r -> r.Market,r.Rl)
            loop ms
        else
            ms' |> List.map (fun r -> r.Market,r.Rl)
    loop ms

let mutable _ps = Unchecked.defaultof<_>

let runAgents parms p ms = 
    (ms,[1..parms.Epochs])
    ||> List.fold(fun ms i ->
        let ms' = runEpisodes  parms p ms
        let avgLoss = ms' |> List.map (fun (_,r) -> r.AvgLoss) |> List.average
        let actDist = 
            ms' 
            |> List.collect (fun (_,r) -> r.Actions) 
            |> List.countBy (fun ar -> ar.Action,ar.IsRand ) 
            |> List.sortBy fst
        let actDstStr = actDist |> List.map (fun ((a,r),c) -> $"{if r then '!' else ' '}{a}:{c}") |> String.concat ","
        printfn $"run {parms.RunId} {i} done, Avg loss:{avgLoss}; Dist:{actDstStr}"
        let ms' = ms' |> List.map(fun (m,r) -> m, {r with Episode = i})
        ms' 
        |> List.tryHead 
        |> Option.iter(fun (_,r0) -> 
                if r0.Episode % 5 = 0 then
                    Test.evalModel parms "current" parms.DQN.Model.Online |> ignore) 

        let ms'' = ms' |> List.map (fun (m,s) -> m, RLState.Reset s)
        ms'')

let trainMarkets parms : (MarketSlice*RLState) list =
    let episodes = Data.dataTrain.Length / EPISODE_LENGTH    
    let idxs = [0 .. episodes-1] |> List.map (fun i -> i * EPISODE_LENGTH)
    idxs
    |> List.map(fun i -> 
        let endIdx = i + EPISODE_LENGTH - 1
        if endIdx <= i then failwith $"Invalid index {i}"
        let mslice = {Market = Data.pricesTrain; StartIndex=i; EndIndex=endIdx}
        let s1 = RLState.Default i 1.0 1000000
        let s = {s1 with Episode = 0; Step = {s1.Step with ExplorationRate = parms.DQN.Exploration.Min}}
        mslice,s)       

let startReRun parms = 
    async {
        try 
            let p = Policy.policy parms
            let ms = trainMarkets parms
            let ps = runAgents parms p ms
            _ps <- ps
        with ex -> printfn "%A" (ex.Message,ex.StackTrace)
    }

let parms1 id (lr,layers)  = 
    let emsize = 32
    let dropout = 0.1
    let max_seq = LOOKBACK
    let nheads = 4
    let nlayers = layers

    let createModel() = 
        let proj = torch.nn.Linear(INPUT_DIM,emsize)
        let ln = torch.nn.LayerNorm(emsize)
        let pos_encoder = PositionalEncoder.create dropout emsize max_seq
        let encoder_layer = torch.nn.TransformerEncoderLayer(emsize,nheads,emsize,dropout)
        let transformer_encoder = torch.nn.TransformerEncoder(encoder_layer,nlayers)        
        let sqrtEmbSz = (sqrt (float emsize)).ToScalar()
        let projOut = torch.nn.Linear(emsize,ACTIONS)
        let activation = torch.nn.Tanh()
        let initRange = 0.1
        let mdl = 
            F [] [proj; pos_encoder; transformer_encoder; projOut; ln]  (fun t -> //B x S x 5
                use p1 = proj.forward(t) // B x S x emsize
                use p = ln.forward(p1)
                use pB2 = p.permute(1,0,2) //batch second - S x B x emsize
                //use mask = Masks.generateSubsequentMask (t.size().[1]) t.device // S x S
                use src = pos_encoder.forward(pB2 * sqrtEmbSz) //S x B x emsize
                use enc = transformer_encoder.call(src) //S x B x emsize
                use encB = enc.permute(1,0,2)  //batch first  // B x S x emsize
                use dec = encB.[``:``,LAST,``:``]    //keep last value as output to compare with target - B x emsize
                use pout = projOut.forward(dec) //B x ACTIONS
                let act = activation.forward pout
                let t_act = Tensor.getDataNested<float32> act
                act
            )
        mdl
    let model = DQNModel.create createModel
    let exp = {Decay=0.9995; Min=0.01; WarupSteps=1000000}
    let DQN = DQN.create model 0.99999f exp ACTIONS device
    {Parms.Default createModel DQN lr id with 
        SyncEverySteps = 3000
        BatchSize = 10
        Epochs = 1000}

let lrs = [0.00001,2L]//; 0.001,8L; 0.001,10]///; 0.0001; 0.0002; 0.00001]
let parms = lrs |> List.mapi (fun i lr -> parms1 i lr)
let restartJobs = parms |> List.map(fun p -> Policy.loadModel p device |> Option.defaultValue p) |> List.map startReRun
 
Test.clearModels()
Data.resetLogs()
restartJobs |> Async.Parallel |> Async.Ignore |> Async.RunSynchronously

(*
verbosity <- LoggingLevel.H
verbosity <- LoggingLevel.M
verbosity <- LoggingLevel.L
verbosity <- LoggingLevel.Q

Test.runTest()

async {Test.evalModels p1} |> Async.Start
(fst _ps).sync (snd _ps)

Policy.model.Online.Module.save @"e:/s/tradestation/temp.bin" 

let m2 = DQN.DQNModel.load Policy.createModel  @"e:/s/tradestation/temp.bin" 

Policy.model.Online.Module.parameters() |> Seq.iter (printfn "%A")

m2.Online.Module.parameters() |> Seq.iter (printfn "%A")

let p1 = m2.Online.Module.parameters() |> Seq.head |> Tensor.getDataNested<float32>
let p2 = Policy.model.Online.Module.parameters() |> Seq.head |> Tensor.getDataNested<float32>
p1 = p2
*)



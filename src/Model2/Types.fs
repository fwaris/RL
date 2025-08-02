﻿module Types
open System
open System.IO
open TsData
open TorchSharp.Fun
open TorchSharp
open FSharpx.Collections
open DQN
open RL

let ( @@ ) a b = Path.Combine(a,b)
let EPISODE_LENGTH = 288/2 // 288 5 min. bars  = 24 hours
let WARMUP = 5000
let EPOCHS = 50
let TREND_WINDOW_BARS_DFLT = 60
let REWARD_HORIZON_BARS = 10
let LOOKBACK_DFLT = int64 (TREND_WINDOW_BARS_DFLT / 2) // 30L
let TX_COST_CNTRCT = 1.0
let MAX_TRADE_SIZE = 1.
let INITIAL_CASH = 5000.0
let INPUT_DIM = 11L
let TRAIN_FRAC = 0.7
let ACTIONS = 3 //0,1,2 - buy, sell, hold
let device = if torch.cuda_is_available() then torch.CUDA else torch.CPU
let data_dir = System.Environment.GetEnvironmentVariable("DATA_DRIVE")
let root = data_dir @@ @"s\tradestation\model1"
let inputDir = data_dir @@ @"s\tradestation"
let INPUT_FILE = inputDir @@ "mes_hist_td2.csv"
//let INPUT_FILE = inputDir @@ "mes_hist_td.csv"
let ensureDirForFilePath (file:string) = 
    let dir = Path.GetDirectoryName(file)
    if dir |> Directory.Exists |> not then Directory.CreateDirectory(dir) |> ignore

let ensureDir (dir:string) = 
    if Directory.Exists dir |> not then Directory.CreateDirectory(dir) |> ignore

type LoggingLevel = Q | L | M | H 
    with  
        member this.IsLow = match this with L | M | H -> true | _ -> false
        member this.isHigh = match this with H -> true | _ -> false
        member this.IsMed = match this with M | H -> true | _ -> false

let mutable verbosity = LoggingLevel.Q

type NBar =
    {
        Freq1 : float
        Freq2 : float
        TrendShort : float
        TrendMed : float
        TrendLong : float
        NOpen  : float 
        NHigh  : float
        NLow   : float
        NClose : float
        NVolume : float
        Bar  : Bar
    }

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
type TuneParms =  
    {
        GoodBuyInterReward      : float
        BadBuyInterPenalty      : float
        ImpossibleBuyPenalty    : float
        GoodSellInterReward     : float
        BadSellInterPenalty     : float
        ImpossibleSellPenalty   : float
        NonInvestmentPenalty    : float
        Layers                  : int64
        Lookback                : int64
        TrendWindowBars         : int
        SkipBars                : int option
        TakeBars                : int option
    }
    with 
        (*
        static member Default = //0.34,-0.84,-0.57,0.98,-0.16,0,0,10
                                //a.GoodBuyInterReward, a.BadBuyInterPenalty, a.ImpossibleBuyPenalty, a.GoodSellInterReward, a.BadSellInterPenalty, a.ImpossibleSellPenalty, a.NonInvestmentPenalty, a.Layers
                        {
                            GoodBuyInterReward = 0.34
                            BadBuyInterPenalty = -0.77
                            ImpossibleBuyPenalty = -0.057
                            GoodSellInterReward = 0.76
                            BadSellInterPenalty = -0.24
                            ImpossibleSellPenalty = 0.0
                            NonInvestmentPenalty = 0.0
                            Layers = 7L
                            Lookback = LOOKBACK
                        }
        *)
        static member Default = //0.34,-0.84,-0.57,0.98,-0.16,0,0,10
                                //a.GoodBuyInterReward, a.BadBuyInterPenalty, a.ImpossibleBuyPenalty, a.GoodSellInterReward, a.BadSellInterPenalty, a.ImpossibleSellPenalty, a.NonInvestmentPenalty, a.Layers
                        {
                            GoodBuyInterReward = 0.34
                            BadBuyInterPenalty = -0.84
                            ImpossibleBuyPenalty = -0.057
                            GoodSellInterReward = 0.98
                            BadSellInterPenalty = -0.16
                            ImpossibleSellPenalty = 0.0
                            NonInvestmentPenalty = 0.0
                            Layers = 10L
                            Lookback = 30L // LOOKBACK
                            TrendWindowBars = 60//TREND_WINDOW_BARS
                            SkipBars = Some (EPISODE_LENGTH * 10)
                            TakeBars = Some (EPISODE_LENGTH * 200)
                        }                        

type Parms =
    {
        LearnRate        : float
        CreateModel      : unit -> IModel                   //need model creation function so that we can load weights from file
        DQN              : DQN
        LossFn           : Loss<torch.Tensor,torch.Tensor,torch.Tensor>
        Opt              : Lazy<torch.optim.Optimizer>
        LearnEverySteps  : int
        SyncEverySteps   : int
        BatchSize        : int
        Epochs           : int
        RunId            : int
        LogSteps         : bool
        SaveModels       : bool
        TuneParms        : TuneParms
    }
    with 
        static member Default modelFn ddqn lr id = 
            let mps = lazy(ddqn.Model.Online.Module.parameters())
            {
                LearnRate       = lr
                CreateModel     = modelFn
                DQN             = ddqn
                LossFn          = torch.nn.SmoothL1Loss()
                Opt             = lazy(torch.optim.Adam(mps.Value, lr=lr,weight_decay=0.00001) :> _) //optimizer should be created after model is moved to target device
                LearnEverySteps = 3
                SyncEverySteps  = 1000
                BatchSize       = 32
                Epochs          = 6
                RunId           = id
                LogSteps        = true
                SaveModels      = true
                TuneParms       = TuneParms.Default
            }

type AgentStats = {
    Actions : Map<int,ActionResult list>
}
    with static member Default = {Actions = Map.empty;}

type AgentBookEntry = {
    Cash : float
    Stock : float    
    Action : int
    NBar : NBar option
}
//keep track of the information we need to run RL in here
type AgentState =
    {
        AgentId          : int
        TimeStep         : int
        CurrentState     : torch.Tensor
        PrevState        : torch.Tensor
        Step             : Step
        InitialCash      : float
        Stock            : float
        TradeSize        : float
        CashOnHand       : float
        LookBack         : int64
        ExpBuff          : Experience.ExperienceBuffer
        S_reward         : float
        S_gain           : float
        CurrentLoss      : float
        Epoch            : int
        Stats            : AgentStats
        AgentBook        : AgentBookEntry list
    }
    with 
        ///reset for new episode
        static member ResetForMarket x = 
            let a = 
                {x with 
                    TimeStep        = 0
                    CashOnHand      = x.InitialCash
                    Stock           = 0
                    AgentBook       = []
                    CurrentState    = torch.zeros([|x.LookBack;INPUT_DIM|],dtype=Nullable torch.float32)
                    PrevState       = torch.zeros([|x.LookBack;INPUT_DIM|],dtype=Nullable torch.float32)
                }            
            // if verbosity.IsLow then 
            //     printfn  $"Reset market called {x.AgentId} exp. rate = {x.Step.ExplorationRate} step = {a.Step.Num}"
            a

        static member ResetForEpisode x = 
            let a =
                { AgentState.ResetForMarket x with 
                    Stats = AgentStats.Default
                }
            //if verbosity.isHigh then 
            //    printfn  $"Reset market called {x.AgentId} exp. rate = {x.Step.ExplorationRate} step = {a.Step.Num}"
            a

        static member Default agentId initExpRate initialCash tp = 
            let expBuff = Experience.createStratifiedSampled (int 50e3) 5
            {
                TimeStep         = 0
                AgentId          = agentId
                CurrentState     = torch.zeros([|tp.Lookback;INPUT_DIM|],dtype=Nullable torch.float32)
                PrevState        = torch.zeros([|tp.Lookback;INPUT_DIM|],dtype=Nullable torch.float32)
                Step             = {ExplorationRate = initExpRate; Num=0}
                Stock            = 0
                TradeSize        = 0.0
                CashOnHand       = initialCash
                InitialCash      = initialCash
                LookBack         = tp.Lookback
                ExpBuff          = expBuff
                S_reward         = -1.0
                S_gain           = -1.0
                Epoch            = 0
                CurrentLoss      = 0.0
                Stats            = AgentStats.Default
                AgentBook        = []
            }


type StepResult = {Market:MarketSlice; Rl:AgentState; ActionResult:ActionResult}

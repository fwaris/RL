module Types
open System
open System.IO
open TsData
open TorchSharp.Fun
open TorchSharp
open FSharpx.Collections
open DQN
open RL

let ( @@ ) a b = Path.Combine(a,b)

let TREND_WINDOW_BARS = 20
let REWARD_HORIZON_BARS = 10
let LOOKBACK = 10L
let TX_COST_CNTRCT = 0.5
let INITIAL_CASH = 100000.0
let MAX_TRADE_SIZE = 25.
let EPISODE_LENGTH = 336 //* 5
let INPUT_DIM = 6L
let TRAIN_FRAC = 0.7
let ACTIONS = 3 //0,1,2 - buy, sell, hold
let device = if torch.cuda_is_available() then torch.CUDA else torch.CPU
let data_dir = System.Environment.GetEnvironmentVariable("DATA_DRIVE")
let root = data_dir @@ @"s\tradestation\model1"
let inputDir = data_dir @@ @"s\tradestation"
let INPUT_FILE = inputDir @@ "mes_hist_td2.csv"

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
            }

type AgentStats = {
    Actions : Map<int,ActionResult list>
}
    with static member Default = {Actions = Map.empty;}

//keep track of the information we need to run RL in here
type AgentState =
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
        ExpBuff          : Experience.ExperienceBuffer
        S_reward         : float
        S_gain           : float
        CurrentLoss      : float
        Epoch            : int
        Stats            : AgentStats
    }
    with 
        ///reset for new episode
        static member ResetForMarket x = 
            let a = 
                {x with 
                    //Step            = {x.Step with Num=0} //keep current exploration rate; just update step number
                    TimeStep        = 0
                    CashOnHand      = x.InitialCash
                    Stock           = 0
                    State           = torch.zeros([|x.LookBack;INPUT_DIM|],dtype=Nullable torch.float32)
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
            if verbosity.IsLow then 
                printfn  $"Reset episode called {x.AgentId} exp. rate = {x.Step.ExplorationRate} step = {a.Step.Num}"
            a

        static member Default agentId initExpRate initialCash = 
            let expBuff = Experience.createStratifiedSampled (int 3e5) 2
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
                Epoch          = 0
                CurrentLoss      = 0.0
                Stats            = AgentStats.Default
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

type StepResult = {Market:MarketSlice; Rl:AgentState; ActionResult:ActionResult}

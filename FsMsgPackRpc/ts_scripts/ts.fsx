#load "../scripts/packages.fsx"
#load "../TsData.fs"
#load "../RL.fs"
open System.Threading.Tasks
open TorchSharp
open TorchSharp.Fun
open TsData
open FSharpx.Collections
open RL

let fn = @"E:\s\tradestation\mes_5_min.bin"

let data = TsData.loadBars fn

type RLState =
    {
        State           : torch.Tensor
        TimeStep        : int
        Stock           : int
        CashOnHand      : float
        InitialCash     : float
        LookBack        : int64
    }
    with 
        static member Default initialCash = 
            let lookback = 40L
            {
                State           = torch.zeros([|lookback;5L|],dtype=torch.float32)
                TimeStep        = 0
                Stock           = 0
                CashOnHand      = initialCash
                InitialCash     = initialCash
                LookBack        = lookback
            }

type Market = {prices : Bar array}
    with member this.IsDone t = t >= this.prices.Length 

module Agent = 
    let bar (env:Market) t = if env.prices.Length < t then env.prices.[t] |> Some else None
    let avgPrice bar = 0.5 * (bar.High - bar.Low)        

    let buy (env:Market) (s:RLState) = 
        bar env s.TimeStep
        |> Option.map (fun bar -> 
            let avgP = avgPrice bar
            let newStock = s.CashOnHand / avgP |> floor
            let cash = if newStock > 0 then s.CashOnHand - (newStock * avgP) else s.CashOnHand
            let stock = s.Stock + (int newStock)
            {s with CashOnHand=cash; Stock=stock})
        |> Option.defaultValue s

    let sell (env:Market) (s:RLState) =
        bar env s.TimeStep
        |> Option.map (fun bar -> 
            let avgP = avgPrice bar
            let newCash = float s.Stock * avgP + s.CashOnHand
            {s with CashOnHand=newCash; Stock=0})
        |> Option.defaultValue s

    let doAction env s act = if act = 0 then buy env s else sell env s

    let skipHead = torch.TensorIndex.Slice(1L)

    let getObservations (env:Market) (s:RLState) =         
        if env.IsDone s.TimeStep then s 
        else                                
            let b =  env.prices.[s.TimeStep]
            let t1 = torch.tensor([|b.Open;b.High;b.Low;b.Close;b.Volume|],dtype=torch.float32)
            let ts = torch.vstack([|s.State;t1|])
            let ts2 = if ts.shape.[0] > s.LookBack then ts.index skipHead else ts  // 40 x 5 
            {s with State = ts2}
        
    let computeRewards env s =         
        let isDone,r = 
            bar env s.TimeStep
            |> Option.map (fun bar -> 
                let avgP = avgPrice  bar
                let total = s.CashOnHand + (float s.Stock * avgP)
                false,total / s.InitialCash
            )
            |> Option.defaultValue (true,0.)
        (s,isDone,r)

    let agent = 
        {
            doAction = doAction
            getObservations = getObservations
            computeRewards = computeRewards
        }


module Policy =
    open DDQN

    let createModel() = 
        torch.nn.Conv1d(40L, 64L, 4L, stride=2L, padding=3L)
        ->> torch.nn.ReLU()
        ->> torch.nn.Conv1d(64L,64L,3L)
        ->> torch.nn.ReLU()
        ->> torch.nn.Flatten()
        ->> torch.nn.Linear(128L,2L)

    let model = DDQNModel.create createModel
    let lossFn = torch.nn.functional.smooth_l1_loss()

    let exp = {Rate = 1.0; Decay=0.999; Min=0.01}
    let expBuff = {Buffer=RandomAccessList.empty; Max=50000}
    let ddqn = DDQN.create model 0.9999f exp 2 torch.CPU
    let batchSize = 32
    let opt = torch.optim.Adam(model.Online.Module.parameters(), lr=0.00025)

    let updateQ td_estimate td_target =
        use loss = lossFn.Invoke(td_estimate,td_target)
        opt.zero_grad()
        loss.backward()
        use t = opt.step() 
        loss.ToDouble()

    let rec policy ddqn buff = 
        {
            selectAction  = fun (s:RLState) -> 
                let act,ddqn' = DDQN.selectAction s.State ddqn
                (policy ddqn buff),act

            update = fun (s:RLState) isDone reward ->                
                let states,nextStates,rewards,actions,dones = Experience.recall batchSize buff  //sample from experience
                use states = states.``to``(ddqn.Device)
                use nextStates = nextStates.``to``(ddqn.Device)
                let td_est = DDQN.td_estimate states actions ddqn        
                //let td_est_d = td_est.data<float32>().ToArray() //ddqn invocations
                let td_tgt = DDQN.td_target rewards nextStates dones ddqn
                let loss = updateQ td_est td_tgt //update online model 
                policy ddqn buff                
        }

    let initPolicy() = policy ddqn expBuff
        
let market = {prices = data}
let run (policy,state) =
    let rec loop (policy,state) =
        if market.IsDone state.TimeStep |> not then
            let p,s = RL.step market Agent.agent (policy,state)
            loop (p,s)
    loop (policy,state)
        
async {run (Policy.initPolicy(), RLState.Default 1000000.)} |> Async.Start


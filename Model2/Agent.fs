module Agent
open DQN
open Experience
open Types
open TorchSharp
open RL

let bar (env:MarketSlice) t = env.Bar t    

let buy (env:MarketSlice) (s:AgentState) = 
    match bar env s.TimeStep with
    | Some nbar ->     
        let avgP = Data.avgPrice nbar.Bar
        let priceWithCost = avgP + TX_COST_CNTRCT
        let stockToBuy = s.CashOnHand / priceWithCost |> floor |> max 0. |> min MAX_TRADE_SIZE
        let outlay = stockToBuy * priceWithCost
        let coh = s.CashOnHand - outlay |> max 0.            
        let stock = s.Stock + stockToBuy 
        assert (stock >= 0.)
        {s with CashOnHand=coh; Stock=stock; TradeSize = stockToBuy}
    | None -> s

let sell (env:MarketSlice) (s:AgentState) =
    match bar env s.TimeStep with 
    | Some nbar -> 
        let avgP = Data.avgPrice nbar.Bar
        let priceWithCost = avgP - TX_COST_CNTRCT
        let stockToSell = s.Stock |> min MAX_TRADE_SIZE
        let inlay = stockToSell * priceWithCost
        let coh = s.CashOnHand + inlay
        let remStock = s.Stock - stockToSell |> max 0.        
        {s with CashOnHand=coh; Stock=remStock; TradeSize = -stockToSell}
    | None -> s 

let doAction _ env s act =
    let book = {Cash = s.CashOnHand; Stock = s.Stock}
    let s = 
        match act with
        | 0 -> buy env s
        | 1 -> sell env s
        | _ -> s                //hold
    let agentBook = if act < 2 then book::s.AgentBook else s.AgentBook //track changes in positions
    {s with TimeStep = s.TimeStep + 1; AgentBook = agentBook}

let skipHead = torch.TensorIndex.Slice(1)

let canBuy avgP s = s.CashOnHand > avgP + TX_COST_CNTRCT
let canSell s = s.Stock > 0
let couldBuy s = s.AgentBook |> List.tryHead |> Option.map (fun b -> b.Stock > s.Stock) |> Option.defaultValue false
let couldSell s = s.AgentBook |> List.tryHead |> Option.map (fun b -> b.Stock > s.Stock) |> Option.defaultValue false

let getObservations _ (env:MarketSlice) (s:AgentState) =         
    let b =  bar env s.TimeStep |> Option.defaultWith (fun () -> failwith "bar not found")
    let avgP = Data.avgPrice b.Bar
    let buySell = torch.tensor([|canBuy avgP s; canSell s|],dtype=torch.float32)
    let t1 = torch.tensor([|b.TrendLong;b.TrendMed;b.TrendShort;b.NOpen;b.NHigh;b.NLow;b.NClose|],dtype=torch.float32)
    let t1 = torch.hstack(buySell,t1)
    let ts = torch.vstack([|s.CurrentState;t1|])
    let ts2 = if ts.shape.[0] > s.LookBack then ts.index skipHead else ts  // LOOKBACK * INPUT_DIM
    {s with CurrentState = ts2; PrevState = s.CurrentState}
        
let computeRewards1 parms env s action =         
    match bar env s.TimeStep with 
    | Some cBar ->
        let avgP     = Data.avgPrice  cBar.Bar
        let sGain    = ((avgP * float s.Stock + s.CashOnHand) - s.InitialCash) / s.InitialCash
        let isDone   = env.IsDone (s.TimeStep + 1)
        let reward = if isDone then sGain else 0.0
        if verbosity.isHigh then
            printfn $"{s.AgentId}-{s.TimeStep}|{s.Step.Num} - P:%0.3f{avgP}, OnHand:{s.CashOnHand}, S:{s.Stock}, R:{reward}, A:{action}, Exp:{s.Step.ExplorationRate} Gain:{sGain}"
        let logLine = $"{s.AgentId},{s.Epoch},{s.TimeStep},{action},{avgP},{s.CashOnHand},{s.Stock},{reward},{sGain},{parms.RunId},{env.StartIndex},{isDone}"
        Data.logger.Post (s.Epoch,parms.RunId,logLine)
        let experience = {NextState = s.CurrentState; Action=action; State = s.PrevState; Reward=float32 reward; Done=isDone }
        let experienceBuff = Experience.append experience s.ExpBuff  
        {s with ExpBuff = experienceBuff; S_reward=reward; S_gain = sGain;},isDone,reward
    | _ -> (s,false,0.0)

let computeRewards parms env s action =         
    match bar env s.TimeStep with 
    | Some cBar ->
        let avgP = Data.avgPrice  cBar.Bar
        let interReward = 
            match action with 
            | 0 when couldBuy s  -> 0.00
            | 0                  -> -1.0
            | 1 when couldSell s -> 0.00
            | 1                  -> -1.0
            | _                  -> if s.Stock <= 0 then -0.001 else 0.0 //if (s.CashOnHand / s.InitialCash) >= 1.0 then  +0.001 else -0.001
        let sGain    = ((avgP * float s.Stock + s.CashOnHand) - s.InitialCash) / s.InitialCash
        let isDone   = env.IsDone (s.TimeStep + 1)
        let reward  = 
            if isDone then 
                sGain
                //if s.Stock = 0 && s.InitialCash = s.CashOnHand then -1.0 else sGain
            else 
                interReward                
        if verbosity.isHigh then
            printfn $"{s.AgentId}-{s.TimeStep}|{s.Step.Num} - P:%0.3f{avgP}, OnHand:{s.CashOnHand}, S:{s.Stock}, R:{reward}, A:{action}, Exp:{s.Step.ExplorationRate} Gain:{sGain}"
        let logLine = $"{s.AgentId},{s.Epoch},{s.TimeStep},{action},{avgP},{s.CashOnHand},{s.Stock},{reward},{sGain},{parms.RunId},{env.StartIndex},{isDone}"
        Data.logger.Post (s.Epoch,parms.RunId,logLine)
        let experience = {NextState = s.CurrentState; Action=action; State = s.PrevState; Reward=float32 reward; Done=isDone }
        let experienceBuff = Experience.append experience s.ExpBuff  
        {s with ExpBuff = experienceBuff; S_reward=reward; S_gain = sGain;},isDone,reward
    | None -> (s,false,0.0)
       
let agent  = 
    {
        doAction = doAction
        getObservations = getObservations
        computeRewards = computeRewards
    }

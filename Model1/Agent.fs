module Agent
open DQN
open Experience
open Types
open TorchSharp
open RL

let bar (env:MarketSlice) t = env.Bar t    

let buy (env:MarketSlice) (s:AgentState) = 
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

let sell (env:MarketSlice) (s:AgentState) =
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

let getObservations _ (env:MarketSlice) (s:AgentState) =         
    let b =  bar env s.TimeStep |> Option.defaultWith (fun () -> failwith "bar not found")
    let t1 = torch.tensor([|b.TrendLong;b.TrendShort;b.NOpen;b.NHigh;b.NLow;b.NClose|],dtype=torch.float32)
    let ts = torch.vstack([|s.State;t1|])
    let ts2 = if ts.shape.[0] > s.LookBack then ts.index skipHead else ts  // 40 x 5             
    {s with State = ts2; PrevState = s.State}
        
let computeRewards parms env s action =         
    bar env s.TimeStep
    |> Option.map (fun cBar -> 
        let avgP     = Data.avgPrice  cBar.Bar
        let sGain    = ((avgP * float s.Stock + s.CashOnHand) - s.InitialCash) / s.InitialCash
        let isDone   = env.IsDone (s.TimeStep + 1)
        let reward  = 
            if isDone then 
                sGain 
            else
                (s.CashOnHand / s.InitialCash) * -0.001
        if verbosity.isHigh then
            printfn $"{s.AgentId}-{s.TimeStep}|{s.Step.Num} - P:%0.3f{avgP}, OnHand:{s.CashOnHand}, S:{s.Stock}, R:{reward}, A:{action}, Exp:{s.Step.ExplorationRate} Gain:{sGain}"
        let logLine = $"{s.AgentId},{s.Epoch},{s.TimeStep},{action},{avgP},{s.CashOnHand},{s.Stock},{reward},{sGain},{parms.RunId},{env.StartIndex},{isDone}"
        Data.logger.Post (s.Epoch,parms.RunId,logLine)
        let experience = {NextState = s.State; Action=action; State = s.PrevState; Reward=float32 reward; Done=isDone }
        let experienceBuff = Experience.append experience s.ExpBuff  
        {s with ExpBuff = experienceBuff; S_reward=reward; S_gain = sGain;},isDone,reward
    )
    |> Option.defaultValue (s,false,0.0)

let computeRewards2 parms env s action =         
    bar env s.TimeStep
    |> Option.map (fun cBar -> 
        let avgP     = Data.avgPrice  cBar.Bar
        let futurePrices = [s.TimeStep .. s.TimeStep + REWARD_HORIZON_BARS] |> List.choose (bar env) |> List.map _.Bar |> List.map Data.avgPrice
        let intermediateReward = 
            if action = 0 && futurePrices |> List.exists (fun p -> p >= avgP + TX_COST_CNTRCT) then
                0.0005
            elif action = 1 && futurePrices |> List.exists (fun p -> p <= avgP - TX_COST_CNTRCT) then
                0.0005
            else
                -0.0005
        let sGain    = ((avgP * float s.Stock + s.CashOnHand) - s.InitialCash) / s.InitialCash
        let isDone   = env.IsDone (s.TimeStep + 1)
        let reward  = 
            if isDone then 
                sGain 
            else 
                //intermediateReward ignore intermediate rewards
                0.0
        if verbosity.isHigh then
            printfn $"{s.AgentId}-{s.TimeStep}|{s.Step.Num} - P:%0.3f{avgP}, OnHand:{s.CashOnHand}, S:{s.Stock}, R:{reward}, A:{action}, Exp:{s.Step.ExplorationRate} Gain:{sGain}"
        let logLine = $"{s.AgentId},{s.Epoch},{s.TimeStep},{action},{avgP},{s.CashOnHand},{s.Stock},{reward},{sGain},{parms.RunId},{env.StartIndex},{isDone}"
        Data.logger.Post (s.Epoch,parms.RunId,logLine)
        let experience = {NextState = s.State; Action=action; State = s.PrevState; Reward=float32 reward; Done=isDone }
        let experienceBuff = Experience.append experience s.ExpBuff  
        {s with ExpBuff = experienceBuff; S_reward=reward; S_gain = sGain;},isDone,reward
    )
    |> Option.defaultValue (s,false,0.0)
       
let agent  = 
    {
        doAction = doAction
        getObservations = getObservations
        computeRewards = computeRewards
    }

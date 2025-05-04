module Agent
open DQN
open Experience
open Types
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

let doAction _ (env:MarketSlice) s act =
    let book = {Cash = s.CashOnHand; Stock = s.Stock; NBar=env.Bar s.TimeStep; Action=act}
    let s = 
        match act with
        | 0 -> buy env s
        | 1 -> sell env s
        | _ -> s                //hold
    let agentBook =  book::s.AgentBook //track changes in positions
    {s with TimeStep = s.TimeStep + 1; AgentBook = agentBook}



let canBuy avgP s = s.CashOnHand > avgP + TX_COST_CNTRCT
let canSell s = s.Stock > 0
let couldBuy s = s.AgentBook |> List.tryHead |> Option.map (fun b -> b.Stock > s.Stock) |> Option.defaultValue false
let couldSell s = s.AgentBook |> List.tryHead |> Option.map (fun b -> b.Stock > s.Stock) |> Option.defaultValue false

let hasBetterPriceForBuy currentPrice futurePrices = futurePrices |> List.exists (fun p -> p > currentPrice + TX_COST_CNTRCT)
let hasBetterPriceForSell currentPrice futurePrices = futurePrices |> List.exists (fun p -> p < currentPrice -  TX_COST_CNTRCT)

let getObservations _ (env:MarketSlice) (s:AgentState) =         
    let b =  bar env s.TimeStep |> Option.defaultWith (fun () -> failwith "bar not found")
    let avgP = Data.avgPrice b.Bar
    let buySell = [|canBuy avgP s; canSell s|] |> Array.map (fun x -> if x then 1.0 else 0.0)
    let st =  [|buySell.[0] ; buySell.[1]; b.Freq1;b.Freq2;b.TrendLong;b.TrendMed;b.TrendShort;b.NOpen;b.NHigh;b.NLow;b.NClose|]
    {s with CurrentState = st; PrevState = s.CurrentState}
        
let computeRewards parms env s action =         
    match bar env s.TimeStep with 
    | Some cBar ->
        let avgP     = Data.avgPrice  cBar.Bar
        let sGain    = ((avgP * float s.Stock + s.CashOnHand) - s.InitialCash) / s.InitialCash
        let isDone   = env.IsDone (s.TimeStep + 1)
        let reward = if isDone then sGain else sGain * 0.1
        if verbosity.isHigh then
            printfn $"{s.AgentId}-{s.TimeStep}|P:%0.3f{avgP}, OnHand:{s.CashOnHand}, S:{s.Stock}, R:{reward}, A:{action},Gain:{sGain}"
        let logLine = $"{s.AgentId},{s.Epoch},{s.TimeStep},{action},{avgP},{s.CashOnHand},{s.Stock},{reward},{sGain},{parms.RunId},{env.StartIndex},{isDone}"
        if parms.LogSteps then
            Data.logger.Post (s.Epoch,parms.RunId,logLine)
        let experience : VExperience.VExperience = {NextState = s.CurrentState; Action=action; State = s.PrevState; Reward=reward; Done=isDone }
        let experienceBuff = VExperience.append experience s.ExpBuff  
        {s with ExpBuff = experienceBuff; S_reward=reward; S_gain = sGain;},isDone,reward
    | _ -> (s,false,0.0)

       
let agent  = 
    {
        doAction = doAction
        getObservations = getObservations
        computeRewards = computeRewards
    }

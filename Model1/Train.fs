module Train 
open System
open System.IO
open Plotly.NET
open TorchSharp
open TorchSharp.Fun
open DQN
open RL
open Types

let acctBlown (s:AgentState) = s.CashOnHand < 10000.0 && s.Stock <= 0
let isDone (m:MarketSlice,s) = m.IsDone (s.TimeStep+1) || acctBlown s

let addAction marketIndex ar actionMap = 
    actionMap
    |> Map.tryFind marketIndex 
    |> Option.map (fun xs -> actionMap |> Map.add marketIndex (ar::xs))
    |> Option.defaultWith (fun _ -> actionMap |> Map.add marketIndex [ar])

//single step a single agent in the given market
let stepAgent parms plcy (m,s) = 
    let s',ar = step parms m Agent.agent plcy s 
    let s'' = 
        {s' with // update step number and exploration rate and collect action result (for stats and reporting)
            Step = DQN.updateStep parms.DQN.Exploration s'.Step; Stats.Actions=addAction m.StartIndex ar s.Stats.Actions
        } 
    let plcy,s'' = if s''.Step.Num % parms.LearnEverySteps = 0 then plcy.update parms s'' else plcy,s''
    if s''.Step.Num % parms.SyncEverySteps = 0 then plcy.sync parms s''    
    {Market=m; Rl=s''; ActionResult=ar}

let rec runAgentOnMarket parms plcy market agent = 
    let rslt = stepAgent parms plcy (market,agent)
    if rslt.ActionResult.IsDone then 
        rslt.Rl
    else
        runAgentOnMarket parms plcy market rslt.Rl

let reportEpisode parms (agent:AgentState) = 
    let actDstStr = 
        agent.Stats.Actions
        |> Map.toSeq 
        |> Seq.collect snd
        |> Seq.countBy (fun ar -> ar.ActionTaken,ar.IsRand ) 
        |> Seq.toList
        |> List.sortBy fst
        |> List.map (fun ((a,r),c) -> $"""{if r then $"!{a}" else $"{a}"}:{c}""")
        |> String.concat ","
    printfn $"ParmId:{parms.RunId}, Id:{agent.AgentId}, Epoch:{agent.Epoch}, Loss:%0.4f{agent.CurrentLoss}, Dist:{actDstStr}, Gain:%0.3f{agent.S_gain}"            
        
          
//run agent once across all market slices
let trainEpisode  parms plcy (agent:AgentState) (ms:MarketSlice list) =
    let ms = List.randomShuffle ms                                  //shuffle market slices to randomize training
    let agent' = (agent,ms) ||> List.fold (fun agent market ->
        let agent' = runAgentOnMarket parms plcy market agent
        AgentState.ResetForMarket agent')
    parms.Scheduler.Value.step()
    reportEpisode parms agent'    
    agent'


let trainEpisodes parms plcy ms = 
    let agent = AgentState.Default 0 1.0 INITIAL_CASH parms.TuneParms
    (agent,[1..parms.Epochs])
    ||> List.fold (fun agent e -> 
        let agent' = AgentState.ResetForEpisode agent
        let agent'' = trainEpisode parms plcy {agent' with Epoch=e} ms        
        Policy.saveModel parms agent''
        agent'')

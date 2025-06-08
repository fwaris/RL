module Policy
open System
open System.IO
open TorchSharp
open TorchSharp.Fun
open DQN
open RL
open Types

let private updateQOnline parms state = 
    let device = DQNModel.device parms.DQN.Model
    let states,nextStates,rewards,actions,dones = Experience.recall device parms.BatchSize state.ExpBuff  //sample from experience
    use states = states
    use nextStates = nextStates
    use td_est = DQN.td_estimate states actions parms.DQN.Model.Online   //online qvals of state-action pairs
    use td_tgt = DQN.td_target rewards nextStates dones parms.DQN   //discounted qvals of opt-action next states
    let loss = parms.LossFn.forward(td_est,td_tgt)
    use u_loss = loss.mean()
    let avgLoss = u_loss.ToDouble()
    parms.Opt.Value.zero_grad()
    loss.backward()
    if verbosity.IsLow && state.Step.Num % 1000 = 0 then 
        printfn $"ParmsId: {parms.RunId}, Step {state.Step.Num} / Epoch: {state.Epoch} : MB Loss: %0.3f{avgLoss}"
        printfn $"Actions"
        let t_td_est = Tensor.getData<float32> td_tgt
        Seq.zip actions t_td_est 
        |> Seq.chunkBySize 5
        |> Seq.iter (fun xs ->
            xs |> Seq.iter (fun (a,v) -> printf $"{a} %0.3f{v} "); printfn "")
    if false (*avgLoss |> Double.IsNaN*) then 
        let t_states = Tensor.getDataNested<float32> states
        let t_td_est = Tensor.getDataNested<float32> td_est
        let t_td_tgt = Tensor.getDataNested<float32> td_tgt
        ()
    parms.Opt.Value.step() |> ignore
    {state with CurrentLoss = avgLoss}

let loadModel parms =
    let dir = root @@ "models_restart"
    ensureDir dir
    Directory.GetFiles(dir,$"model_{parms.RunId}*") |> Seq.sortByDescending (fun f -> (FileInfo f).LastWriteTime) |> Seq.tryHead
    |> Option.map(fun fn ->
        let mdl  = DQN.DQNModel.load parms.CreateModel fn                        
        let dqn = {parms.DQN with Model = mdl; }
        {parms with DQN = dqn})

let private syncModel parms s = 
    DQNModel.sync parms.DQN.Model

let saveModel parms s = 
    let fn = root @@ "models" @@ $"model_{parms.RunId}_{s.Epoch}_{s.Step.Num}.bin"
    ensureDirForFilePath fn
    if parms.SaveModels then
        DQNModel.save fn parms.DQN.Model 
    if verbosity.IsLow then printfn "Synced"

let rec policy parms = 
    {
        selectAction = fun parms (s:AgentState) -> 
            let act,isRandom =  DQN.selectAction s.CurrentState parms.DQN s.Step
            act,isRandom

        update = fun parms state  ->      
            if state.Step.Num < parms.DQN.Exploration.WarupSteps then 
                policy parms, state
            else
                let state' = updateQOnline parms state
                policy parms, state'

        sync = syncModel
    }

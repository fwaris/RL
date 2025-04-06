module Policy
open System
open System.IO
open TorchSharp
open TorchSharp.Fun
open DQN
open RL
open Types

let private updateQ parms (lossTensor:torch.Tensor) =        
    lossTensor.backward()
    let avgLoss = lossTensor.mean().ToDouble()
    use t = parms.Opt.step()         
    if Double.IsNaN avgLoss then
        let pns = parms.DQN.Model.Online.Module.named_parameters() |> Seq.map(fun struct(n,x) -> n, Tensor.getDataNested<float32> x) |> Seq.toArray
        ()
        failwith "Nan loss"
    avgLoss

let private loss parms s = 
    parms.Opt.zero_grad()
    let states,nextStates,rewards,actions,dones = Experience.recall parms.BatchSize s.ExpBuff  //sample from experience
    use states = states.``to``(parms.DQN.Device)
    use nextStates = nextStates.``to``(parms.DQN.Device)
    let td_est = DQN.td_estimate states actions parms.DQN           //estimate the Q-value of state-action pairs from online model
    let td_tgt = DQN.td_target rewards nextStates dones parms.DQN   //
    let loss = parms.LossFn.forward(td_est,td_tgt)
    if verbosity.IsLow && s.Step.Num % 1000 = 0 then 
        printfn $"Step {s.Step.Num}"
        printfn $"Actions"
        printfn "%A" actions
        let t_td_est = Tensor.getDataNested<float32> td_tgt
        printfn $"Esimate Q vals"
        printfn "%A" t_td_est

    if loss.ToDouble() |> Double.IsNaN then 
        let t_states = Tensor.getDataNested<float32> states
        let t_nextStates = Tensor.getDataNested<float32> nextStates
        let t_states = Tensor.getDataNested<float32> states
        let t_td_est = Tensor.getDataNested<float32> td_est
        let t_td_tgt = Tensor.getDataNested<float32> td_tgt
        ()
    loss

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


let private syncModel parms s = 
    DQNModel.sync parms.DQN.Model parms.DQN.Device
    let fn = root @@ "models" @@ $"model_{parms.RunId}_{s.Epoch}_{s.Step.Num}.bin"
    ensureDirForFilePath fn
    DQNModel.save fn parms.DQN.Model 
    if verbosity.IsLow then printfn "Synced"

let rec policy parms = 
    {
        selectAction = fun parms (s:AgentState) -> 
            let act,isRandom =  DQN.selectAction s.State parms.DQN s.Step
            act,isRandom

        update = fun parms state  ->      
            let lossTensor = loss parms state
            let avgLoss = updateQ parms lossTensor
            if Double.IsNaN avgLoss then
                let ls1 = lossTensor |> Tensor.getData<float32>
                ()
            if verbosity.IsMed then printfn $"avg loss {avgLoss}"
            let state' = {state with CurrentLoss=avgLoss}
            policy parms, state'

        sync = syncModel
    }

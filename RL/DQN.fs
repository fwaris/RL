///abstractions for training a model under the DQN regime
///based on https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
module DQN
open TorchSharp
open TorchSharp.Fun
open System
open FSharpx.Collections
open FSharp.Collections.ParallelSeq

///DDQNModel maintains target and online versions of a model
type DDQNModel = {Target : IModel;  Online : IModel}

type Exploration = {Decay:float; Min:float; WarupSteps:int} with static member Default = {Decay = 0.999; Min=0.01; WarupSteps = 1000}
type Step = {Num:int; ExplorationRate:float}
type DQN = {Model:DDQNModel; Gamma:float32; Exploration:Exploration; Actions:int; }

module DQNModel =
    let device (models:DDQNModel) = 
        models.Online.Module.parameters() 
        |> Seq.tryHead 
        |> Option.map _.device
        |> Option.defaultWith (fun _ -> failwith "unable to get torch device")

    let create (fmodel: unit -> IModel) = 
        let tgt = fmodel()
        let onln = fmodel()
        tgt.Module.parameters() |> Seq.iter (fun p -> p.requires_grad <- false)
        {Target=tgt; Online=onln}
        
    let sync models =
        models.Target.Module.load_state_dict(models.Online.Module.state_dict())
        |> ignore

    let save (file:string) (ddqn:DDQNModel)  = ddqn.Online.Module.save(file) |> ignore

    let load (fmodel:unit -> IModel) (file:string) =
        let ddqn = create fmodel
        try
            ddqn.Online.Module.load(file) |> ignore
            ddqn.Target.Module.load(file) |> ignore
        with ex -> 
            printfn $"invalid model file {file} - returning empty model"
            printfn "%A" (ex.Message,ex.StackTrace)
        ddqn

    let dispose (ddqn:DDQNModel) =
        ddqn.Online.Module.Dispose()
        ddqn.Target.Module.Dispose()


module DQN =
    //use randomization from single source - pytorch
    let rand() : float = torch.rand([|1L|],dtype=Nullable torch.double).item()
    let randint n : int = torch.randint(n,[|1|],dtype=torch.int32).item()

    let updateStep exp step =
        let expRate = 
            if step.Num <= exp.WarupSteps
                then step.ExplorationRate 
                else step.ExplorationRate * exp.Decay |> max exp.Min
        {
            Num = step.Num + 1
            ExplorationRate = expRate
        }

    let create model gamma exploration actions =
        {
            Model = model
            Exploration = exploration
            Gamma = gamma
            Actions = actions
        }

    let selectAction (state:torch.Tensor) ddqn step =
        let actionIdx =
            if rand() < step.ExplorationRate then //explore
                randint ddqn.Actions,true
            else
                let device = DQNModel.device ddqn.Model
                use state = state.``to``(device)  //exploit
                use state = state.unsqueeze(0)
                use action_values = ddqn.Model.Online.forward(state)
                use maxQAct = action_values.argmax()
                maxQAct.ToInt32(),false
        actionIdx

    let actionIdx (actions:torch.Tensor) = 
        [|
            torch.TensorIndex.Tensor (torch.arange(actions.shape.[0], dtype=torch.int64))  //batch dimension
            torch.TensorIndex.Tensor (actions)                                             //actions dimension
        |]

    let td_estimate (state:torch.Tensor) (actions:int[]) (model:IModel) =
        let s = state.detach_()
        let s = s.requires_grad_(true)
        use q = model.forward(s)                                   //value of each available actions (when taken from the give state)
        let idx = actionIdx (torch.tensor(actions,dtype=torch.int64))  //indexes of the actions actually taken by agents (in the given batch)
        let actVals = q.index(idx)                                      //values of the taken actions

        if false then //set to true to debug
            let t_state = Tensor.getDataNested<float32> state
            let t_q = Tensor.getDataNested<float32> q
            let t_actVals = Tensor.getDataNested<float32> actVals
            ()
            
        actVals
    
    ///Find the best action for the given state from the online model.
    ///Then find the q-value of that action (for the same state) from the target model.
    ///Use the target q-value for the discounted reward for model update.
    ///Since target is lagged, as per DDQN this, stabilizes the q values (otherwise the model can 'chase its own tail')
    let td_target (reward:float32[]) (next_state:torch.Tensor) (isDone:bool[]) ddqn =
        use t = torch.no_grad()                              //turn off gradient calculation
        use q_online = ddqn.Model.Online.forward(next_state) //online model estimate of value (from next state)
        use best_action = q_online.argmax(1L)                //optimum value action from online

        let idx = actionIdx best_action                      //index of optimum value action

        use q_target      = ddqn.Model.Target.forward(next_state) //target model estimates of value (from next state)
        use q_target_best = q_target.index(idx)                   //value of best action according to target model 
                                                                  //where the 'best action' is determined by the online model
        let device = DQNModel.device ddqn.Model
        use d_reward' = torch.tensor(reward)
        use d_reward = d_reward'.``to``(device)  //reward to device (cpu/gpu)
        use d_isDone' = torch.tensor(isDone)
        use d_isDone = d_isDone'.``to``(device)  //was episode done?
        use d_isDoneF = d_isDone.float()                         //convert boolean to float32
        use ret = d_reward + (1.0f.ToScalar() -  d_isDoneF) * ddqn.Gamma.ToScalar() * q_target_best //reward + discounted value

        if false then //set to true to debug
            let t_d_isDoneF = Tensor.getDataNested<float32> d_isDoneF
            let t_q_online = Tensor.getDataNested<float32> q_online
            let t_best_action = Tensor.getDataNested<int64> best_action
            let t_q_target_best = Tensor.getDataNested<float32> q_target_best
            let t_ret = Tensor.getDataNested<float32> ret
            let i = 1
            ()

        ret.float()   //convert to float32

    
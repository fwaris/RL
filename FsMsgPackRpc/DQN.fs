///abstractions for training a model under the DQN regime
///based on https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
module DQN
open TorchSharp
open TorchSharp.Fun
open System.IO
open FSharpx.Collections

///DDQNModel maintains target and online versions of a model
type DDQNModel = {Target : IModel;  Online : IModel}
type Experience = {State:torch.Tensor; NextState:torch.Tensor; Action:int; Reward:float32; Done:bool}
type ExperienceBuffer = {Buffer : RandomAccessList<Experience>; Max:int}
type Exploration = {Decay:float; Min:float} with static member Default = {Decay = 0.999; Min=0.01}
type Step = {Num:int; ExplorationRate:float}
type DQN = {Model:DDQNModel; Gamma:float32; Exploration:Exploration; Actions:int; Device:torch.Device }

module DQNModel =
    let create (fmodel: unit -> IModel) = 
        let tgt = fmodel()
        let onln = fmodel()
        tgt.Module.parameters() |> Seq.iter (fun p -> p.requires_grad <- false)
        {Target=tgt; Online=onln}

    let sync models (device:torch.Device) =
        let onlineMdl = models.Online.Module.cpu()
        let tgtMdl = models.Target.Module.cpu()
        let dict = onlineMdl.state_dict()
        tgtMdl.load_state_dict(dict) |> ignore
        onlineMdl.``to``(device) |> ignore
        tgtMdl.``to``(device) |> ignore

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

module Experience =
    let createBuffer maxExperiance = {Buffer =RandomAccessList.empty; Max=maxExperiance}

    let append exp buff = 
        let ls = buff.Buffer
        let ls = RandomAccessList.cons exp ls
        let ls =
            if RandomAccessList.length ls > buff.Max then 
                RandomAccessList.uncons ls |> snd
            else
                ls
        {buff with Buffer = ls}

    let sample n buff =
        if buff.Buffer.Length <= n then
            buff.Buffer |> Seq.toArray 
        else
            let idx = torch.randperm(int64 buff.Buffer.Length,dtype=torch.int) |> Tensor.getData<int> 
            [|for i in 0..n-1 -> buff.Buffer.[idx.[i]]|]

    let recall n buff =
        let exps = sample n buff
        let states     = exps |> Array.map (fun x->x.State.unsqueeze(0L)) |> torch.vstack
        let nextStates = exps |> Array.map (fun x->x.NextState.unsqueeze(0L)) |> torch.vstack
        let actions    = exps |> Array.map (fun x->x.Action)
        let rewards    = exps |> Array.map (fun x -> x.Reward)
        let dones      = exps |> Array.map (fun x->x.Done)
        states,nextStates,rewards,actions,dones

    type Tser = int*int64[]*List<float32[]*float32[]*int*float32*bool> //use simple types for serialization
    let save path buff =
        let data = 
            buff.Buffer 
            |> Seq.map (fun x-> 
                x.State.data<float32>().ToArray(),
                x.NextState.data<float32>().ToArray(),
                x.Action,
                x.Reward,
                x.Done
            )
            |> Seq.toList

        if Seq.isEmpty data then failwithf "empty buffer cannot be saved as tensor shape is unknown"
        let shape = (Seq.head buff.Buffer).State.shape
        let ser = MBrace.FsPickler.BinarySerializer()
        use str = System.IO.File.Create (path:string)
        let sval:Tser = (buff.Max,shape,data)
        ser.Serialize(str,sval)

    let load path =
        let ser = MBrace.FsPickler.BinarySerializer()
        use str = System.IO.File.OpenRead(path:string)        
        let ((mx,shape,data):Tser) = ser.Deserialize<Tser>(str)
        let buff = createBuffer mx    
        let buff = 
            (buff,data)
            ||> Seq.fold (fun acc (st,nst,act,rwd,dn) -> 
                let exp =
                    {
                        State       = torch.tensor(st,dimensions=shape)
                        NextState   = torch.tensor(nst, dimensions=shape)
                        Action      = act
                        Reward      = rwd
                        Done        = dn
                    }
                append exp acc)
        buff

module DQN =
    //use randomization from single source - pytorch
    let rand() : float = torch.rand([|1L|],dtype=torch.double).item()
    let randint n : int = torch.randint(n,[|1|],dtype=torch.int32).item()

    let updateStep exp step =
        {
            Num = step.Num + 1
            ExplorationRate = step.ExplorationRate * exp.Decay |> max exp.Min
        }

    let create model gamma exploration actions (device:torch.Device) =
        model.Target.Module.``to``(device) |> ignore
        model.Online.Module.``to``(device) |> ignore
        {
            Model = model
            Exploration = exploration
            Gamma = gamma
            Actions = actions
            Device = device
        }

    let selectAction (state:torch.Tensor) ddqn step =
        let actionIdx =
            if rand() < step.ExplorationRate then //explore
                randint ddqn.Actions
            else
                use state = state.``to``(ddqn.Device)  //exploit
                use state = state.unsqueeze(0)
                use action_values = ddqn.Model.Online.forward(state)
                action_values.argmax().ToInt32()
        actionIdx

    let actionIdx (actions:torch.Tensor) = 
        [|
            torch.TensorIndex.Tensor (torch.arange(actions.shape.[0], dtype=torch.int64))  //batch dimension
            torch.TensorIndex.Tensor (actions)                                             //actions dimension
        |]

    let td_estimate (state:torch.Tensor) (actions:int[]) ddqn =
        use q = ddqn.Model.Online.forward(state)                         //value of each available action (when taken from the give state)
        let idx = actionIdx (torch.tensor(actions,dtype=torch.int64))    //index of action actually taken 
        q.index(idx)                                                     //value of taken action

    let td_target (reward:float32[]) (next_state:torch.Tensor) (isDone:bool[]) ddqn =
        use t = torch.no_grad()                              //turn off gradient calculation
        use q_online = ddqn.Model.Online.forward(next_state) //online model estimate of value (from next state)
        use best_action = q_online.argmax(dimension=1L)      //optimum value action from online
        let idx = actionIdx best_action                      //index of optimum value action

        use q_target      = ddqn.Model.Target.forward(next_state) //target model estimates of value (from next state)
        use q_target_best = q_target.index(idx)                   //value of best action according to target model 
                                                                  //where the 'best action' is determined by the online model

        use t_reward = torch.tensor(reward).``to``(ddqn.Device)  //reward to device (cpu/gpu)
        use t_isDone = torch.tensor(isDone).``to``(ddqn.Device)  //was episode done?
        use t_isDoneF = t_isDone.float()                         //convert boolean to float32
        use ret = t_reward + (1.0f.ToScalar() -  t_isDoneF) * ddqn.Gamma.ToScalar() * q_target_best //reward + discounted value
        ret.float()                                                                                 //convert to float32









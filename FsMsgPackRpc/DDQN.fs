///abstractions for training a model under the DDQN regime
///based on https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
module DDQN
open TorchSharp
open TorchSharp.Fun
open System.IO
open FSharpx.Collections

///DDQNModel maintains target and online versions of a model
type DDQNModel = {Target : IModel;  Online : IModel}
type Experience = {State:torch.Tensor; NextState:torch.Tensor; Action:int; Reward:float32; Done:bool}
type ExperienceBuffer = {Buffer : RandomAccessList<Experience>; Max:int}
type Exploration = {Rate:float; Decay:float; Min:float}
type Step = {Step:int; ExplorationRate:float}
type DDQN = {Model:DDQNModel; Gamma:float32; Exploration:Exploration; Step:Step; Actions:int; Device:torch.Device }

module DDQNModel =
    let create (fmodel: unit -> IModel) = 
        let tgt = fmodel()
        let onln = fmodel()
        tgt.Module.parameters() |> Seq.iter (fun p -> p.requires_grad <- false)
        {Target=tgt; Online=onln}

    let sync models =
        models.Target.Module.load_state_dict(models.Online.Module.state_dict()) |> ignore

    let save file ddqn  = ddqn.Online.Module.Save(file)

    let load (fmodel:unit -> IModel) file =
        let ddqn = create fmodel
        let mdl = torch.nn.Module.Load(file)
        ddqn.Online.Module.load_state_dict(mdl.state_dict()) |> ignore
        ddqn.Target.Module.load_state_dict(mdl.state_dict()) |> ignore
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

module DDQN =
    //use randomization from single source - pytorch
    let rand() : float = torch.rand([|1L|],dtype=torch.double).item()
    let randint n : int = torch.randint(n,[|1|],dtype=torch.int32).item()

    let private updateStep (ddqn:DDQN) =
        let step' =
            {
                Step = ddqn.Step.Step + 1
                ExplorationRate = ddqn.Step.ExplorationRate * ddqn.Exploration.Decay |> max ddqn.Exploration.Min
            }
        {ddqn with Step = step'}

    let create model gamma exploration actions (device:torch.Device) =
        model.Target.Module.``to``(device) |> ignore
        model.Online.Module.``to``(device) |> ignore
        {
            Model = model
            Exploration = exploration
            Gamma = gamma
            Step = {Step=0; ExplorationRate=exploration.Rate}
            Actions = actions
            Device = device
        }

    let selectAction (state:torch.Tensor) (ddqn:DDQN) =
        let actionIdx =
            if rand() < ddqn.Step.ExplorationRate then //explore
                randint ddqn.Actions
            else
                use state = state.``to``(ddqn.Device)  //exploit
                use state = state.unsqueeze(0)
                use action_values = ddqn.Model.Online.forward(state)
                action_values.argmax().ToInt32()
        actionIdx,updateStep ddqn 

    let td_estimate (state:torch.Tensor) (actions:int[]) ddqn =
        use q = ddqn.Model.Online.forward(state) //batch x actions
        let idx = [|torch.TensorIndex.Single 0; torch.TensorIndex.Tensor ( torch.tensor(actions,dtype=torch.int64))|]
        q.index(idx)

    let td_target (reward:float32[]) (next_state:torch.Tensor) (isDone:bool[]) ddqn =
        use t = torch.no_grad()
        use t_reward = torch.tensor(reward).``to``(ddqn.Device)
        use t_isDone = torch.tensor(isDone).``to``(ddqn.Device)
        use t_isDoneF = t_isDone.float()
        use next_state_q = ddqn.Model.Online.forward(next_state)
        use best_action  = next_state_q.argmax(dimension=1L)
        use next_qs = ddqn.Model.Target.forward(next_state)
        let idx = [|torch.TensorIndex.Single 0; torch.TensorIndex.Tensor(best_action)|]
        use next_q = next_qs.index(idx)
        use ret = t_reward + (1.0f.ToScalar() -  t_isDoneF) * ddqn.Gamma.ToScalar() * next_q
        use _ = torch.enable_grad()
        ret.float()









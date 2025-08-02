module Policy
open System
open System.IO
open RL
open Types
open FsCgp
open FsCgp.CgpBase
open FsCgp.CgpRun
open System.Threading
open VDQN
open TorchSharp
open System.Numerics

let softmax (xs:float[]) =
    let exps = xs |> Array.map exp
    let sumExp = Array.sum exps
    exps |> Array.map (fun v -> v / sumExp)

let isAllZero (tensor: torch.Tensor) : bool =
    tensor.eq(0.f.ToScalar()).all().ToBoolean()

let funcs =
  [|
    (fun (xs:torch.Tensor[]) -> xs.[0]),2, "x"
    (fun (xs:torch.Tensor[]) -> xs.[1]),2, "y"
    (fun (xs:torch.Tensor[]) -> xs.[0] + xs.[1]),2,"add"
    (fun (xs:torch.Tensor[]) -> xs.[0] - xs.[1]),2,"subtract"
    (fun (xs:torch.Tensor[]) -> xs.[0] * xs.[1]),2,"multiply"
    (fun (xs:torch.Tensor[]) -> if isAllZero xs.[1] then xs.[1] else xs.[0] / xs.[1]),2,"division"
    (fun (xs:torch.Tensor[]) -> xs.[0].max(xs.[1])),2,"max"
    (fun (xs:torch.Tensor[]) -> xs.[0].min(xs.[1])),2,"min"
    (fun (xs:torch.Tensor[]) -> xs.[0].sin()),1,"sin"
    (fun (xs:torch.Tensor[]) -> xs.[0].cos()),1,"cos"
    (fun (xs:torch.Tensor[]) -> xs.[0].tan()),1,"tan"
    (fun (xs:torch.Tensor[]) -> xs.[0].tanh()),1,"tanh"
    (fun (xs:torch.Tensor[]) -> xs.[0].log()),1,"log"
    (fun (xs:torch.Tensor[]) -> xs.[0].absolute_()),1,"absolute"
    (fun (xs:torch.Tensor[]) -> xs.[0].sigmoid_()),1,"sigmoid"
    (fun (xs:torch.Tensor[]) -> xs.[0].silu_()),1,"silu"
    (fun (xs:torch.Tensor[]) -> xs.[0].bernoulli_()),1,"bernouli"
    (fun (xs:torch.Tensor[]) -> xs.[0].cholesky_inverse()),1,"cholesky inverse"
  |]

let ft = funcs |> Array.map (fun (f,a,d) -> {F=f;Arity=a;Desc=d})


let tensorConst() = 
  {
    NumConstants = 1
    Max = torch.tensor(100.f)
    ConstGen = fun() -> torch.randint(100,[|1|],dtype=torch.float32)
    Evolve = fun i -> torch.randn([|1L|]) + i
  }

(*
  ///utility function to create ConstSpec for floats
  let floatConsts numConst maxConst = 
    {
      NumConstants = numConst
      Max = maxConst
      ConstGen = fun() -> 
        let sign = if Probability.RNG.Value.NextDouble() > 0.5 then 1.0 else -1.0
        let v = Probability.RNG.Value.NextDouble() * maxConst
        v * sign //|> int |> float
      Evolve = fun i -> 
        let sign = if Probability.RNG.Value.NextDouble() > 0.5 then 1.0 else -1.0
        let v = Probability.RNG.Value.NextDouble()
        i + (sign * v) //|> int |> float
    }*)

let spec = 
  {
    NumInputs = INPUT_DIM
    NumNodes = 30
    NumOutputs = 3
    BackLevel = None
    FunctionTable = ft
    MutationRate = 0.20
    Constants = Some (tensorConst())
  }


//points fitting f(x) = x³ - 2x + 10.
let test_cases =
  [|
        (0., 10.)
        (0.5, 9.125)
        (1., 9.)
        (10., 990.)
        (-5., -105.)
        (17., 4889.)
        (3.14, 34.679144)
  |]
  |> Array.map (fun (inp,out)-> [|inp|],[|out|])

let loss (y':float[]) (y:float[]) = (y'.[0] - y.[0]) ** 2.0 //square loss y' is output from the genome evaluation and y is actual output 

let cspec = compile spec

let evaluator = createEvaluator cspec loss

let termination gen loss = gen > 100000

let currentBest = ref Unchecked.defaultof<_>

let cts = new  CancellationTokenSource()
let obsTestCases,fps = Observable.createObservableAgent cts.Token None //observable to send new data to learner

let runAsync() =
  async {
    do run1PlusLambdaDynamic 
        Verbose cspec 10  evaluator obsTestCases 
        termination (fun indv -> currentBest.Value <- indv) None
  }
  |> Async.Start

let showBest() = callGraph cspec currentBest.Value.Genome //|> visualize

let postTests() = fps test_cases //sends (new or updated) data to let the learner dynamically adapt to changing data  



let private updateQOnline parms state = 
    let states,nextStates,rewards,actions,dones = VExperience.recall parms.BatchSize state.ExpBuff  //sample from experience    
    let td_est = VDQN.td_estimate states actions parms.VDQN.Model.Online   //online qvals of state-action pairs
    let td_tgt = VDQN.td_target rewards nextStates dones parms.VDQN   //discounted qvals of opt-action next states


    let loss = parms.LossFn.forward(td_est,td_tgt)
    let avgLoss = loss.mean().ToDouble()
    parms.Opt.Value.zero_grad()
    loss.backward()
    parms.Opt.Value.step() |> ignore
    if verbosity.IsLow && state.Step.Num % 1000 = 0 then 
        printfn $"Step {state.Step.Num} / {state.Epoch}"
        printfn $"Actions"
        let t_td_est = Tensor.getData<float32> td_tgt
        Seq.zip actions t_td_est 
        |> Seq.chunkBySize 5
        |> Seq.iter (fun xs ->
            xs |> Seq.iter (fun (a,v) -> printf $"{a} %0.3f{v} "); printfn "")
    if true (*avgLoss |> Double.IsNaN*) then 
        let t_states = Tensor.getDataNested<float32> states
        let t_nextStates = Tensor.getDataNested<float32> nextStates
        let t_td_est = Tensor.getDataNested<float32> td_est
        let t_td_tgt = Tensor.getDataNested<float32> td_tgt
        let i = 1
        ()
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

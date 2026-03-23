#r "nuget: FSharp.Collections.ParallelSeq, 1.2.0"
#r "nuget: FSharpx.Collections, 3.1.0"
#r "nuget: FsPickler, 5.3.2"
#r "nuget: MathNet.Numerics, 5.0.0"
#r "nuget: Plotly.NET, 5.1.0"
#r "nuget: TorchSharp.Fun, 0.91.0"
#r "nuget: TorchSharp-cuda-windows, 0.106.0"

#load "src/RL/DQN.fs"
#load "src/RL/VExperience.fs"

open System
open System.IO
open Plotly.NET
open TorchSharp
open TorchSharp.Fun
open DQN
open VExperience

// Run with:
//   dotnet fsi cart_pole.fsx

type CartPoleConfig =
    {
        Gravity: float
        MassCart: float
        MassPole: float
        Length: float
        ForceMag: float
        Tau: float
        XThreshold: float
        ThetaThresholdRadians: float
        MaxEpisodeSteps: int
    }

type CartPoleState =
    {
        Observation: float32[]
        Steps: int
    }

module CartPole =
    let config =
        {
            Gravity = 9.8
            MassCart = 1.0
            MassPole = 0.1
            Length = 0.5
            ForceMag = 10.0
            Tau = 0.02
            XThreshold = 2.4
            ThetaThresholdRadians = 12.0 * Math.PI / 180.0
            MaxEpisodeSteps = 500
        }

    let private totalMass = config.MassCart + config.MassPole
    let private poleMassLength = config.MassPole * config.Length

    let reset (rng: Random) =
        let obs =
            Array.init 4 (fun _ ->
                let sample = rng.NextDouble() * 0.1 - 0.05
                float32 sample)
        { Observation = obs; Steps = 0 }

    let step action state =
        let x = float state.Observation.[0]
        let xDot = float state.Observation.[1]
        let theta = float state.Observation.[2]
        let thetaDot = float state.Observation.[3]

        let force =
            if action = 1 then config.ForceMag
            else -config.ForceMag

        let costheta = cos theta
        let sintheta = sin theta

        // Standard CartPole dynamics update, matching the classic-control equations.
        let temp =
            (force + poleMassLength * thetaDot * thetaDot * sintheta) / totalMass

        let thetaAcc =
            (config.Gravity * sintheta - costheta * temp) /
            (config.Length * (4.0 / 3.0 - config.MassPole * costheta * costheta / totalMass))

        let xAcc =
            temp - poleMassLength * thetaAcc * costheta / totalMass

        let nextX = x + config.Tau * xDot
        let nextXDot = xDot + config.Tau * xAcc
        let nextTheta = theta + config.Tau * thetaDot
        let nextThetaDot = thetaDot + config.Tau * thetaAcc
        let nextSteps = state.Steps + 1

        let isDone =
            nextX < -config.XThreshold ||
            nextX > config.XThreshold ||
            nextTheta < -config.ThetaThresholdRadians ||
            nextTheta > config.ThetaThresholdRadians ||
            nextSteps >= config.MaxEpisodeSteps

        let reward = 1.0f

        {
            Observation =
                [|
                    float32 nextX
                    float32 nextXDot
                    float32 nextTheta
                    float32 nextThetaDot
                |]
            Steps = nextSteps
        },
        reward,
        isDone

let seed = 123
let rng = Random(seed)
let evalRng = Random(seed + 1)
torch.random.manual_seed(int64 seed) |> ignore

let tryGetIntArg name =
    fsi.CommandLineArgs
    |> Array.tryFindIndex ((=) name)
    |> Option.bind (fun idx ->
        if idx + 1 < fsi.CommandLineArgs.Length then
            Int32.TryParse(fsi.CommandLineArgs.[idx + 1]) |> function
            | true, value -> Some value
            | _ -> None
        else
            None)

let episodes = tryGetIntArg "--episodes" |> Option.defaultValue 350
let batchSize = 64
let replayCapacity = 10000
let learningStarts = 128
let targetSyncEvery = 250
let gamma = 0.99f
let learningRate = 1e-4
let evalEvery = 10

let createModel () =
    torch.nn.Linear(4L, 128L)
    ->> torch.nn.ReLU()
    ->> torch.nn.Linear(128L, 128L)
    ->> torch.nn.ReLU()
    ->> torch.nn.Linear(128L, 2L)

// The library keeps separate online and target networks for Double DQN updates.
let ddqn = DQNModel.create createModel
let device = torch.CPU
ddqn.Online.Module.``to``(device) |> ignore
ddqn.Target.Module.``to``(device) |> ignore

let exploration =
    {
        DQN.Exploration.Default with
            Decay = 0.999
            Min = 0.05
            WarupSteps = learningStarts
    }

let dqn = DQN.create ddqn gamma exploration 2
let optimizer = torch.optim.AdamW(dqn.Model.Online.Module.parameters(), lr = learningRate, amsgrad = true)
let lossFn = torch.nn.SmoothL1Loss()

let optimizeModel replayBuffer =
    let states, nextStates, rewards, actions, dones = VExperience.recall batchSize replayBuffer
    // Replay samples come back as float32[] states, so flatten then reshape into [batch, features].
    use statesTensor =
        torch.tensor(Array.collect id states, dtype = torch.float32, device = device)
            .reshape(int64 states.Length, 4L)
    use nextStatesTensor =
        torch.tensor(Array.collect id nextStates, dtype = torch.float32, device = device)
            .reshape(int64 nextStates.Length, 4L)
    // Estimate Q(s,a) with the online network and bootstrap targets with the lagged target network.
    let tdEstimate = DQN.td_estimate statesTensor actions dqn.Model.Online
    let tdTarget = DQN.td_target rewards nextStatesTensor dones dqn
    use loss = lossFn.forward(tdEstimate, tdTarget)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(dqn.Model.Online.Module.parameters(), 100.0) |> ignore
    optimizer.step() |> ignore
    loss.ToSingle()

let evaluate episodes =
    [|
        for _ in 1 .. episodes do
            let mutable state = CartPole.reset evalRng
            let mutable done_ = false
            let mutable totalReward = 0.0f

            while not done_ do
                use stateTensor = torch.tensor(state.Observation, dtype = torch.float32, device = device)
                let action, _ = DQN.bestAction stateTensor dqn
                let nextState, reward, isDone = CartPole.step action state
                totalReward <- totalReward + reward
                state <- nextState
                done_ <- isDone

            yield totalReward
    |]

let mutable replayBuffer = VExperience.createUniformSampled replayCapacity
let mutable stepInfo = { DQN.Step.Num = 0; ExplorationRate = 0.9 }
let recentRewards = Collections.Generic.Queue<float32>()
let episodeRewards = Collections.Generic.List<float>()
let rollingRewards = Collections.Generic.List<float>()
let evalEpisodes = Collections.Generic.List<int>()
let evalMeans = Collections.Generic.List<float>()

for episode in 1 .. episodes do
    let mutable state = CartPole.reset rng
    let mutable done_ = false
    let mutable episodeReward = 0.0f
    let mutable latestLoss = None

    while not done_ do
        use stateTensor = torch.tensor(state.Observation, dtype = torch.float32, device = device)
        let action, wasRandom = DQN.selectAction stateTensor dqn stepInfo
        let nextState, reward, isDone = CartPole.step action state

        replayBuffer <-
            VExperience.append
                {
                    State = state.Observation
                    NextState = nextState.Observation
                    Action = action
                    Reward = reward
                    Done = isDone
                    Priority = 1.0f
                }
                replayBuffer

        if replayBuffer.Length() >= learningStarts && replayBuffer.Length() >= batchSize then
            latestLoss <- Some (optimizeModel replayBuffer)

        // Periodically copy online weights into the target network for more stable targets.
        if stepInfo.Num > 0 && stepInfo.Num % targetSyncEvery = 0 then
            DQNModel.sync dqn.Model

        stepInfo <- DQN.updateStep dqn.Exploration stepInfo
        episodeReward <- episodeReward + reward
        state <- nextState
        done_ <- isDone

        if wasRandom && stepInfo.Num % 250 = 0 then
            printfn $"step={stepInfo.Num} exploration=%.3f{stepInfo.ExplorationRate} buffer={replayBuffer.Length()}"

    recentRewards.Enqueue episodeReward
    if recentRewards.Count > 20 then
        recentRewards.Dequeue() |> ignore

    let averageReward = recentRewards |> Seq.averageBy float
    episodeRewards.Add(float episodeReward)
    rollingRewards.Add(averageReward)
    let lossText =
        latestLoss
        |> Option.map (fun loss -> $" loss=%.4f{loss}")
        |> Option.defaultValue ""

    printfn
        $"episode={episode} reward=%.1f{episodeReward} avg20=%.1f{averageReward} epsilon=%.3f{stepInfo.ExplorationRate}{lossText}"

    if episode % evalEvery = 0 then
        let evalMean =
            evaluate 10
            |> Array.averageBy float
        evalEpisodes.Add(episode)
        evalMeans.Add(evalMean)
        printfn $"  greedy-eval mean(10)=%.1f{evalMean}"

let evalRewards = evaluate 10
let evalMean = evalRewards |> Array.averageBy float
printfn $"evaluation mean reward over 10 episodes: %.1f{evalMean}"

let episodeAxis = [ 1 .. episodeRewards.Count ]
let rewardChart =
    Chart.Line(episodeAxis, episodeRewards, Name = "Episode reward")
let rollingChart =
    Chart.Line(episodeAxis, rollingRewards, Name = "Rolling avg (20)")
let evalChart =
    Chart.Line(evalEpisodes, evalMeans, Name = "Greedy eval mean (10)")

let trainingChart =
    [ rewardChart; rollingChart; evalChart ]
    |> Chart.combine
    |> Chart.withTitle "CartPole DQN training"
    |> Chart.withXAxisStyle "Episode"
    |> Chart.withYAxisStyle "Reward"

let chartPath = Path.Combine(__SOURCE_DIRECTORY__, "cart_pole_training.html")
trainingChart |> Chart.saveHtml chartPath
printfn $"saved training plot to: {chartPath}"
trainingChart |> Chart.show

DQNModel.dispose dqn.Model

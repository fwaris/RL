# RL

`RL` is an F# reinforcement learning repository centered on Deep Q-Network (DQN) training for discrete action spaces.

The core library lives in `src/RL` and provides reusable building blocks for:

- defining environments and agent interaction loops
- training Double DQN models
- working with replay buffers
- saving and loading replay memory
- experimenting with both tensor-based and value-array-based Q-learning flows

The repo also contains sample projects and scripts that use the library in both minimal and simulator-driven environments.

## Cart Pole Example:

### [`cart_pole.fsx`](/cart_pole.fsx)

A minimal CartPole DQN example inspired by the PyTorch CartPole tutorial, but implemented with this repo's `DQN` and `VExperience` modules. It is a good starting point if you want to see the library on a compact discrete-control problem before moving on to the larger AirSim samples.

`RL` is based on [TorchSharp.Fun](https://github.com/fwaris/TorchSharp.Fun) (which wraps [TorchSharp](https://github.com/dotnet/TorchSharp)) such that models can be defined a function-composition style. The base DQN model for cart pole is given below:
```F#
let createModel () =
    torch.nn.Linear(4L, 128L)
    ->> torch.nn.ReLU()
    ->> torch.nn.Linear(128L, 128L)
    ->> torch.nn.ReLU()
    ->> torch.nn.Linear(128L, 2L)
```

The fastest way to see the library in action is the CartPole script at `cart_pole.fsx`.

Run it from the repository root:

```powershell
dotnet fsi cart_pole.fsx
```

The script:

- trains a small Double DQN agent on the classic CartPole control task
- logs training reward and rolling average
- runs periodic greedy evaluation episodes
- saves an HTML chart to `cart_pole_training.html`

Example training curve from the sample:

![CartPole training curve](/docs/images/cart_pole_training.jpeg)




## What is in the repo

### `src/RL`

The reusable library project. This is the main packageable component of the solution.

Key modules:

- `RL.fs`: core agent and policy abstractions, plus the step function for environment interaction
- `DQN.fs`: Double DQN training helpers built around `TorchSharp` and `TorchSharp.Fun`
- `Experience.fs`: tensor-based replay buffers with uniform, stratified, and prioritized sampling
- `VDQN.fs`: a lighter value-based DQN variant that works on `float[]` state/value functions
- `VExperience.fs`: replay buffer support for the value-based path
- `SeqUtils.fs`: supporting utilities and timeseries/data helpers

### `src/CarSimulator`

Unreal engine based environment and simulator used for RL training.

### `src/CarTrain`

A training application that exercises the RL library against the car simulator.

### `src/RoadMap`

An additional app/project for visualizing the road network coordinate points in the Unreal simulator environment.


## Core ideas

This repo is built around the standard DQN training loop:

1. observe the current state
2. choose an action with epsilon-greedy exploration
3. apply the action to the environment
4. collect reward and terminal status
5. append the transition to replay memory
6. sample minibatches from replay memory
7. update the online Q-network using Double DQN targets
8. periodically sync the target network

The implementation supports:

- epsilon decay with warmup
- online and target network separation
- masked action selection for valid-action subsets
- uniform replay sampling
- stratified replay sampling
- prioritized replay sampling

## Tech stack

- F#
- .NET
- TorchSharp
- TorchSharp.Fun
- MathNet.Numerics
- FsPickler

## Build

From the repository root:

```powershell
dotnet build RL.sln
```

To pack the reusable library project:

```powershell
dotnet pack src/RL/RL.fsproj -c Release
```

## AirSim car sample setup

The car RL sample in this repository is built around Microsoft AirSim running the `Neighborhood` environment, which is distributed in the AirSim releases as `AirSimNH.zip`.

### Download the environment

1. Open the AirSim releases page:
   `https://github.com/microsoft/AirSim/releases`
2. Download `AirSimNH.zip`
3. Extract it to a local folder

AirSim's precompiled-binaries docs also point users to the latest GitHub release for Unreal environments.

### Configure AirSim for cars

This repo includes AirSim helper files under `airsim_commands`:

- `airsim_commands/settings.json`
- `airsim_commands/run.bat`
- `airsim_commands/run_headlesss.bat`

The included settings configure `SimMode` as `Car` and define two PhysX cars (`Car1` and `Car2`).

### Use the repo-provided commands

After extracting `AirSimNH.zip`, copy these files from this repo into the extracted AirSim folder, next to the `AirSimNH.exe` binary:

- `airsim_commands/settings.json`
- `airsim_commands/run.bat`
- `airsim_commands/run_headlesss.bat`

Then start AirSim using one of the included scripts:

- `run.bat`: launches `AirSimNH` in a window at `800x400`
- `run_headlesss.bat`: launches `AirSimNH` with `-RenderOffScreen`

If you prefer, you can also keep `settings.json` elsewhere and point AirSim at it manually, but the simplest repo-aligned setup is to place these files beside the AirSim binary and use the included batch files.

## Who this is for

This repo is a good fit if you want:

- an F#-first DQN library for discrete action problems
- replay buffer implementations you can reuse in your own projects
- a TorchSharp-based starting point for reinforcement learning experiments
- example projects that show how the library can be wired into concrete environments

## License

This repository is licensed under the MIT License. See `LICENSE` for details.

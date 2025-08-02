module VDQN
open System
open FSharpx.Collections


///DDQNModel maintains target and online versions of a model
type VDDQNModel = {Target : Ref<float[]->float[]>;  Online : Ref<float[]->float[]>}

type Exploration = {Decay:float; Min:float; WarupSteps:int} with static member Default = {Decay = 0.999; Min=0.01; WarupSteps = 1000}
type Step = {Num:int; ExplorationRate:float}
type VDQN = {Model:VDDQNModel; Gamma:float; Exploration:Exploration; Actions:int;}

module VDQNModel =
        
    let sync models = models.Target.Value <- models.Online.Value


module VDQN =
    let rng = System.Random()
    let rand() = rng.NextDouble()
    let randint (numActions:int) = rng.Next(numActions)

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

    let argmax (fs:float[]) = fs |> Seq.index |> Seq.maxBy snd |> fst

    let selectAction (state:float[]) vddqn step =
        let actionIdx =
            if rand() < step.ExplorationRate then //explore
                randint vddqn.Actions,true
            else
                let action_values = vddqn.Model.Online.Value(state)
                argmax action_values,false
        actionIdx

    let td_estimate (states:float[][]) (actions:int[]) (model:Ref<float[]->float[]>) =
        Seq.zip states actions 
        |> Seq.map (fun (s,a) -> model.Value(s).[a])
        |> Seq.toArray    

    let td_target (rewards:float[]) (next_states:float[][]) (isDones:bool[]) vddqn =
        let bestActionsOnline = 
            next_states
            |> Array.map (fun s -> vddqn.Model.Online.Value(s) |> argmax)

        let qvals =                                                           //qvals of online best actions by target model
            Array.zip next_states bestActionsOnline
            |> Array.map (fun (s,a) -> (vddqn.Model.Target.Value s).[a])

        let discountVals =
            Seq.zip3 rewards isDones qvals
            |> Seq.map(fun (r,isDone,q_target_best) -> r + (if isDone then 0.0 else vddqn.Gamma) * q_target_best)
            |> Seq.toArray

        discountVals


    
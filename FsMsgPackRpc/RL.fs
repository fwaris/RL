module RL
open System.Threading.Tasks

type Agent<'env,'state> = 
    {
        doAction        : 'env -> 'state -> int -> 'state
        getObservations : 'env -> 'state -> 'state
        computeRewards  : 'env -> 'state -> 'state*bool*float
    }

type Env<'env,'state> =
    {
        reset : 'env -> 'state -> unit
    }

type Policy<'state> =
    {
        selectAction : 'state -> Policy<'state>*int
        update : 'state -> bool -> float -> Policy<'state>
    }

let step env agent (policy,s0) =   
    let policy,act        = policy.selectAction s0
    let s1                = agent.doAction env s0 act
    let s2                = agent.getObservations env s1
    let s3,isDone,reward  = agent.computeRewards env s2 
    let policy            = policy.update s3 isDone reward
    (policy,s3)
    



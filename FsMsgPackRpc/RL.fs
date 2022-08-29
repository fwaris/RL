module RL

type Agent<'parms,'env,'state> = //note: 'parms are hyper parameters that are not expected to change during the run
    {
        doAction        : 'parms -> 'env -> 'state -> int -> 'state                         
        getObservations : 'parms -> 'env -> 'state -> 'state
        computeRewards  : 'parms -> 'env -> 'state -> int -> 'state*bool*float
    }

type Policy<'parms,'state> =
    {
        selectAction : 'parms  -> 'state -> Policy<'parms,'state>*int
        update       : 'parms -> 'state -> bool -> float -> Policy<'parms,'state>*'state
        sync         : 'parms -> 'state -> unit
    }

let step parms env agent (policy,s0) =   
    let policy,act        = policy.selectAction parms s0
    let s1                = agent.doAction parms env s0 act
    let s2                = agent.getObservations parms env s1
    let s3,isDone,reward  = agent.computeRewards parms env s2 act
    let policy,s4         = policy.update parms s3 isDone reward
    (policy,s4)
    


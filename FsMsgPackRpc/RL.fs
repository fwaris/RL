module RL

type Agent<'env,'state> = 
    {
        doAction        : 'env -> 'state -> int -> 'state
        getObservations : 'env -> 'state -> 'state
        computeRewards  : 'env -> 'state -> int -> 'state*bool*float
    }

type Policy<'state> =
    {
        selectAction : 'state -> Policy<'state>*int
        update : 'state -> bool -> float -> Policy<'state>*'state
        sync  :  'state -> unit
    }

let step env agent (policy,s0) =   
    let policy,act        = policy.selectAction s0
    let s1                = agent.doAction env s0 act
    let s2                = agent.getObservations env s1
    let s3,isDone,reward  = agent.computeRewards env s2 act
    let policy,s4         = policy.update s3 isDone reward
    (policy,s4)
    



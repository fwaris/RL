//RL framework for discrete action spaces
module RL

type Agent<'parms,'env,'state> = //note: 'parms are hyper parameters that are not expected to change during the run
    {
        doAction        : 'parms -> 'env -> 'state -> int -> 'state                         
        getObservations : 'parms -> 'env -> 'state -> 'state
        computeRewards  : 'parms -> 'env -> 'state -> int -> 'state*bool*float
    }

type ActionResult = {Action:int; IsDone:bool; Reward:float; IsRand:bool}

///policy can receive a list of state-action-isDone-reward tuples to support multiple agents, each operating in its own environment
type Policy<'parms,'state> =
    {
        selectAction : 'parms -> 'state -> int*bool
        update       : 'parms -> 'state -> Policy<'parms,'state>*'state
        sync         : 'parms -> 'state -> unit
    }

///Single step of an agent in the environment - select action; do action; observe next state; compute reward
let step parms env agent policy s0 =   
    let act,isRand       = policy.selectAction parms s0
    let s1               = agent.doAction parms env s0 act
    let s2               = agent.getObservations parms env s1
    let s3,isDone,reward = agent.computeRewards parms env s2 act
    s3,{Action=act; IsDone=isDone; Reward=reward; IsRand=isRand}
    

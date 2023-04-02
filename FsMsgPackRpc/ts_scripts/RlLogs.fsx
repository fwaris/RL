#load "../scripts/packages.fsx"
open System
open System.IO
open Plotly.NET

module Seq =
    let groupAdjacent f input = 
        let prev = Seq.head input
        let sgs,n = 
            (([],prev),input |> Seq.tail)
            ||> Seq.fold(fun (acc,prev) next -> 
                if f (prev,next) then 
                    match acc with
                    | [] -> [[prev]],next
                    | xs::rest -> (prev::xs)::rest,next
                else
                    match acc with 
                    | [] -> [[];[prev]],next
                    | xs::rest -> []::(prev::xs)::rest,next)
        let sgs = 
            match sgs with 
            | [] -> [[n]]
            | xs::rest -> 
                match xs with
                | [] -> [n]::rest
                | ys -> (n::ys)::rest
        sgs |> List.map List.rev |> List.rev

(*
["a"; "a"; "a"; "b"; "c"; "c"] |> Seq.groupAdjacent (fun (a,b)->a=b)
val it : seq<seq<string>> = seq [["a"; "a"; "a"]; ["b"]; ["c"; "c"]]
*)
let folder = @"e:/s/tradestation/logs"
[folder] |> List.iter(fun d -> if Directory.Exists d |> not then Directory.CreateDirectory d |> ignore)

let allRows() =   
    let files = Directory.GetFiles(folder,"*.csv")
    files 
    |> Seq.collect (fun x -> 
        use str = File.Open(x,FileMode.Open, FileAccess.Read, FileShare.ReadWrite)
        use strw = new StreamReader(str)
        let lines = strw |> Seq.unfold(fun x -> let l = x.ReadLine() in if l <> null then Some (l,x) else None) |> Seq.toArray
        lines)
    |> Seq.toArray

//"agentId,episode,step,action,price,cash,stock,reward,gain,parmId";
type LogE = {
    AgentId : int
    Episode : int
    Step    : int
    Action  : int
    Price   : float
    Cash    : float
    Stock   : int
    Reward  : float
    Gain    : float
    ParmId  : int
}

let toLogE (xs:string[]) =
    try
        {
            AgentId     = int xs.[0]
            Episode     = int xs.[1]
            Step        = int xs.[2]
            Action      = int xs.[3]
            Price       = float xs.[4]
            Cash        = float xs.[5]
            Stock       = int xs.[6]
            Reward      = float xs.[7]
            Gain        = float xs.[8]
            ParmId      = int xs.[9]
        }
        |> Some
    with ex -> 
        printfn "%s" ex.Message
        None
    
let logE() = allRows() |> Array.map(fun l -> l.Split(',')) |> Array.choose toLogE

let a1() = logE() |> Array.filter(fun e -> e.AgentId=0) |> Seq.groupAdjacent (fun (a,b) -> a.Step < b.Step)

let chart() =
    let a1 = a1()
    [0..a1.Length/10..a1.Length-1] |> List.map (fun i -> a1.[i]) |> List.map(fun xs -> xs |> List.map(fun x->x.Gain) |> Chart.Violin) |> Chart.combine |> Chart.show

System.GC.Collect()
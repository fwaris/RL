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
let dataDrive = Environment.GetEnvironmentVariable("DATA_DRIVE")
let folder = @$"{dataDrive}/s/tradestation/logs"
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

let genChart everyNth (f:LogE->float) =
    let a1 = a1()
    printfn $"Len {a1.Length}"
    [0..a1.Length/everyNth..a1.Length-1] 
    |> List.map (fun i -> a1.[i]) 
    |> List.map(fun xs -> xs |> List.map f |> Chart.Violin) 
    |> Chart.combine 
    |> Chart.withTitle $"F distribution over epochs every {everyNth} episode"
    |> Chart.show
    

let gainsChart everyNth =
    let a1 = a1()
    a1.Length
    [0..a1.Length/everyNth..a1.Length-1] 
    |> List.map (fun i -> a1.[i]) 
    |> List.map(fun xs -> xs |> List.map(fun x->x.Gain) |> Chart.Violin) 
    |> Chart.combine 
    |> Chart.withTitle $"Pct gain distribution over epochs every {everyNth} episode"
    |> Chart.show

let cashOnHandChart everyNth =
    let a1 = a1()
    [0..a1.Length/everyNth..a1.Length-1] 
    |> List.map (fun i -> a1.[i]) 
    |> List.map(fun xs -> xs |> List.map(fun x->x.Cash) |> Chart.Violin) 
    |> Chart.combine 
    |> Chart.withTitle $"Cash onhand distribution over epochs every {everyNth} episode"
    |> Chart.show
   


let actionChart everyNth =
    let a1 = a1()
    [0..a1.Length/everyNth..a1.Length-1] 
    |> List.map (fun i -> a1.[i]) 
    |> List.map(fun xs -> xs |> List.map(fun x->x.Action) |> Chart.Violin) 
    |> Chart.combine 
    |> Chart.withTitle $"Action distribution 0=Buy, 1=Sell, 2=Hold: every {everyNth} episode"
    |> Chart.show   

let actionVsChart everyNth (f:LogE->float) =    
    let a1 = a1()
    [0..a1.Length/everyNth..a1.Length-1] 
    |> List.map (fun i -> a1.[i]) 
    |> List.map(fun xs -> Chart.Histogram2D (xs |> List.map(fun x->x.Action), xs |> List.map f))
    |> Chart.combine 
    |> Chart.withTitle $"Action vs X 0=Buy, 1=Sell, 2=Hold: every {everyNth} episode"
    |> Chart.show   

(*
gainsChart 1
actionChart 1
cashOnHandChart 1
*)

let plotGain() = 
    let ag1 = logE()
    let ep = 
        ag1 
        |> Array.countBy _.Episode 
        |> Array.sortByDescending  (fun (x,c) -> c,x) 
        |> Seq.head 
        |> fst
    printfn $"Data length :{ag1.Length}"    
    [
        for a in (ag1 |> Array.distinctBy (fun a -> a.ParmId,a.AgentId)) do
            let ag1 = ag1 |> Array.filter (fun x -> x.AgentId = a.AgentId && x.Episode=ep && x.ParmId = a.ParmId )
            ag1 |> Seq.map _.Gain  |> Seq.indexed |> Chart.Line |> Chart.withTraceInfo $"{a.ParmId}-{a.AgentId}"
    ]
    |> Chart.combine
    |> Chart.withYAxisStyle(MinMax=(-0.1,+0.1))
    |> Chart.withTitle $"Gain ep: {ep}"
    |> Chart.show
    // let ag1 = logE() |> Array.filter (fun x -> x.AgentId = 1 && x.Episode=ep)
    // printfn $"Data length :{ag1.Length}"    
    // ag1 |> Seq.map _.Gain  |> Seq.indexed |> Chart.Line |> Chart.withTitle $"Gain ep: {ep}"
    // |> Chart.show

let test2() =
    plotGain()

    genChart 1 (fun x->float x.Stock)
    genChart 1 (fun x->float x.Reward)
    genChart 1 (fun x->float x.Price)
    actionVsChart 1 (fun x->x.Price)
    actionVsChart 1 (fun x->x.Reward)
    let logs = a1()
    logs |> List.sumBy _.Length
    (List.head logs)

let testPlots() = 
    let ag1 = logE()
    [
    ag1 |> Seq.map _.Cash |> Seq.indexed |> Chart.Line |> Chart.withTraceInfo "Cash"
    ag1 |> Seq.mapi (fun i x -> i, float x.Stock * x.Price) |> Chart.Line |> Chart.withTraceInfo "Stock"
    ]
    |> Chart.combine |> Chart.show

    [
    ag1 |> Seq.map (fun x -> if x.Action=0 then 1 else 0) |> Seq.indexed |> Chart.Line |> Chart.withTraceInfo "buy"
    ag1 |> Seq.map (fun x -> if x.Action=1 then 1 else 0) |> Seq.indexed |> Chart.Line |> Chart.withTraceInfo "sell"
    ag1 |> Seq.map (fun x -> if x.Action=2 then 1 else 0) |> Seq.indexed |> Chart.Line |> Chart.withTraceInfo "hold"
    ]
    |> Chart.combine |> Chart.show

    ag1 |> Seq.mapi (fun i x->i,x.Cash + (float x.Stock * x.Price)) |> Chart.Line |> Chart.show

(*
plotGain()
System.GC.Collect()
*)

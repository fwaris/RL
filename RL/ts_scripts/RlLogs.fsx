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
let model2 = @$"{dataDrive}/s/tradestation/model2/logs"
let model1 = @$"{dataDrive}/s/tradestation/model1/logs"
[model1; model2] |> List.iter(fun d -> if Directory.Exists d |> not then Directory.CreateDirectory d |> ignore)

let allRows folder =   
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
    Market  : int
    IsDone  : bool
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
            Market      = int xs.[10]
            IsDone      = bool.Parse xs.[11]
        }
        |> Some
    with ex -> 
        printfn "%s" ex.Message
        None
    
let logE folder = allRows folder |> Array.map(fun l -> l.Split(',')) |> Array.choose toLogE

let a1 folder = logE folder |> Array.filter(fun e -> e.AgentId=0) |> Seq.groupAdjacent (fun (a,b) -> a.Step < b.Step)

let lastEpoch folder = 
    let ag1 = logE folder
    let ep = 
        ag1 
        |> Array.countBy _.Episode 
        |> Array.sortByDescending  (fun (x,c) -> c,x) 
        |> Seq.head 
        |> fst
    ag1 |> Array.filter (fun x -> x.Episode = ep)


let genChart folder everyNth (f:LogE->float) =
    let a1 = a1 folder
    printfn $"Len {a1.Length}"
    [0..a1.Length/everyNth..a1.Length-1] 
    |> List.map (fun i -> a1.[i]) 
    |> List.map(fun xs -> xs |> List.map f |> Chart.Violin) 
    |> Chart.combine 
    |> Chart.withTitle $"F distribution over epochs every {everyNth} episode"
    |> Chart.show
    

let gainsChart folder everyNth =
    let a1 = a1 folder
    a1.Length
    [0..a1.Length/everyNth..a1.Length-1] 
    |> List.map (fun i -> a1.[i]) 
    |> List.map(fun xs -> xs |> List.map(fun x->x.Gain) |> Chart.Violin) 
    |> Chart.combine 
    |> Chart.withTitle $"Pct gain distribution over epochs every {everyNth} episode"
    |> Chart.show

let cashOnHandChart folder everyNth =
    let a1 = a1 folder
    [0..a1.Length/everyNth..a1.Length-1] 
    |> List.map (fun i -> a1.[i]) 
    |> List.map(fun xs -> xs |> List.map(fun x->x.Cash) |> Chart.Violin) 
    |> Chart.combine 
    |> Chart.withTitle $"Cash onhand distribution over epochs every {everyNth} episode"
    |> Chart.show


let actionChart folder everyNth =
    let a1 = a1 folder
    [0..a1.Length/everyNth..a1.Length-1] 
    |> List.map (fun i -> a1.[i]) 
    |> List.map(fun xs -> xs |> List.map(fun x->x.Action) |> Chart.Violin) 
    |> Chart.combine 
    |> Chart.withTitle $"Action distribution 0=Buy, 1=Sell, 2=Hold: every {everyNth} episode"
    |> Chart.show   

let actionVsChart folder everyNth (f:LogE->float) =    
    let a1 = a1 folder
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

let plotGain folder = 
    let ag1 = logE folder
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
    |> Chart.withTitle $"Gain ep: {ep}<br>{folder}"
    |> Chart.show
    // let ag1 = logE() |> Array.filter (fun x -> x.AgentId = 1 && x.Episode=ep)
    // printfn $"Data length :{ag1.Length}"    
    // ag1 |> Seq.map _.Gain  |> Seq.indexed |> Chart.Line |> Chart.withTitle $"Gain ep: {ep}"
    // |> Chart.show

let plotLastEpicGainByMarket folder = 
    let ag1 = lastEpoch folder
    let ep = ag1 |> Seq.tryHead |> Option.map _.Episode |> Option.defaultValue 0
    printfn $"Data length :{ag1.Length}"    
    ag1 
    |> Array.groupBy _.ParmId
    |> Array.collect(fun (p,xs) ->
        xs 
        |> Array.groupBy _.Market
        |> Array.map(fun (m,ys) -> 
            ys 
            |> Array.sortBy _.Step
            |> Array.map _.Gain
            |> Array.indexed 
            |> Chart.Line 
            |> Chart.withTraceInfo $"Market {m}"
        ))
    |> Chart.combine 
    |> Chart.withTitle $"Gain by market slice. Ep {ep}<br>{folder}"
    |> Chart.show

let plotLastEpicActionsByNMarkets n folder = 
    let ag1 = lastEpoch folder
    let ep = ag1 |> Seq.tryHead |> Option.map _.Episode |> Option.defaultValue 0
    printfn $"Data length :{ag1.Length}"    
    ag1 
    |> Array.groupBy _.ParmId
    |> Array.collect(fun (p,xs) ->
        xs 
        |> Array.groupBy _.Market
        |> Array.take n
        |> Array.collect(fun (m,ys) -> 
            [|
                ys 
                |> Array.sortBy _.Step
                |> Array.indexed 
                |> Array.filter (fun (i,x) -> x.Action = 0)
                |> Array.map (fun (i,x) -> i,x.Price)
                |> Chart.Point
                |> Chart.withMarkerStyle (Symbol=StyleParam.MarkerSymbol.ArrowUp, Size=10)
                |> Chart.withTraceInfo $"Buy Market {m}"
                ys 
                |> Array.sortBy _.Step
                |> Array.indexed 
                |> Array.filter (fun (i,x) -> x.Action = 1)
                |> Array.map (fun (i,x) -> i,x.Price)
                |> Chart.Point
                |> Chart.withMarkerStyle (Symbol=StyleParam.MarkerSymbol.ArrowDown, Size=10)
                |> Chart.withTraceInfo $"Sell Market {m}"                
                ys 
                |> Array.sortBy _.Step
                |> Array.map _.Price
                |> Array.indexed                 
                |> Chart.Line 
                |> Chart.withTraceInfo $"Price Market {m}"
            |]
        ))
    |> Chart.combine 
    |> Chart.withLegendStyle(Visible=false)
    |> Chart.withTitle $"Action vs Price 2 Markets {ep}<br>{folder}"
    |> Chart.withSize(1000.,800.)
    |> Chart.show


let plotGrainTrendByMarket folder = 
    let ag1 = logE folder
    printfn $"Data length :{ag1.Length}"    
    ag1 
    |> Array.groupBy _.ParmId
    |> Array.collect(fun (p,xs) ->
        xs 
        |> Array.groupBy _.Market
        |> Array.map(fun (m,ys) -> 
            ys 
            |> Array.filter _.IsDone
            |> Array.sortBy _.Episode
            |> Array.map _.Gain
            |> Array.indexed 
            |> Chart.Line 
            |> Chart.withTraceInfo $"Market {m}"
        ))
    |> Chart.combine 
    |> Chart.withTitle $"Gains by market slice over Epochs<br>{folder}"
    |> Chart.show


let test2() =
    let folder = model2
    plotGain folder

    genChart folder 1 (fun x->float x.Stock)
    genChart folder 1 (fun x->float x.Reward)
    genChart folder 1 (fun x->float x.Price)
    actionVsChart folder 1 (fun x->x.Price)
    actionVsChart folder 1 (fun x->x.Reward)
    let logs = a1 folder
    logs |> List.sumBy _.Length
    (List.head logs)

let testPlots folder = 
    let ag1 = logE folder
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
plotGrainTrendByMarket model1
plotLastEpicGainByMarket model1
plotLastEpicActionsBy2Markets model2
plotLastEpicActionsByNMarkets 4 model1
plotGrainTrendByMarket model2
plotLastEpicGainByMarket model2
plotLastEpicActionsBy2Markets model2
*)

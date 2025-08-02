#load "../scripts/packages.fsx"
open System
open System.IO
open Plotly.NET
open FSharp.Data
open System.Text.RegularExpressions
open OpenCvSharp
open MachineLearning


let dataDrive = Environment.GetEnvironmentVariable("DATA_DRIVE")

let (|FileNumber|_|) (inp:string) = 
    let m = Regex.Match(inp,"opt_(\d+)\.csv")
    printfn $"{inp} - {if m.Success then m.NextMatch().Groups[1].Value else System.String.Empty}"
    if m.Success then m.Groups.[1].Value |> Some else None

let logFileName (folder:string) =
    Directory.GetFiles(folder,"opt*.csv")
    |> Array.choose(function FileNumber n -> Some n | _ -> None)
    |> Array.map int
    |> Array.sortDescending
    |> Array.tryHead
    |> Option.map (fun n -> $"opt_{n}.csv")
    |> Option.defaultValue $"opt.csv"

let model1 = @$"{dataDrive}/s/tradestation/model1/"
let [<Literal>] INPUT_FILE = (@"e:\s\tradestation\model1\opt.csv")

type T_Log = CsvProvider<INPUT_FILE>

let logFilePath = Path.Combine(model1,logFileName model1)
let t_log = T_Log.Load(logFilePath).Rows |> Seq.toList //|> List.skip 1000

let hist2d (title:string) (f1:T_Log.Row->float) (f2:T_Log.Row->float) (xs:T_Log.Row seq) =
    xs
    |> Seq.map (fun x -> f1 x, f2 x)
    |> Seq.toList
    |> List.unzip
    |> fun (xs,ys) -> Chart.Histogram2D(xs,ys) 
    |> Chart.withTitle title
    |> Chart.show

let scatter (title:string) (f1:T_Log.Row->float) (f2:T_Log.Row->float) (xs:T_Log.Row seq) =
    xs
    |> Seq.map (fun x -> f1 x, f2 x)
    |> Seq.toList
    |> List.unzip
    |> fun (xs,ys) -> Chart.Point(xs,ys) 
    |> Chart.withTitle title
    |> Chart.show

let basis = 0.01
let toVec (r:T_Log.Row) = 
    let trendWindwBars = float r.TendWindowBars * 10.
    [|
        float r.Layers
        (trendWindwBars / 3.0)
        trendWindwBars
        float r.GoodBuyInterReward * basis
        float r.GoodSellInterReward * basis
        float r.BadBuyInterPenalty * basis
        float r.BadSellInterPenalty * basis
        float r.ImpossibleBuyPenalty * basis
        float r.ImpossibleSellPenalty * basis
        float r.NonInvestmentPenalty * basis
    |]

open Microsoft.Diagnostics.NETCore.Client
open System.Diagnostics
let (@@) (a:string) (b:string) = Path.Combine(a,b)
let dumpMemory() =
    Process.GetProcesses() |> Seq.iter (fun p -> printfn $"{p.ProcessName}, {p.Id}")
    let processId = 
        Process.GetProcessesByName("Model1") 
        |> Seq.tryHead 
        |> Option.map _.Id 
        |> Option.defaultWith (fun _ -> failwith "process not found")
    let client = new DiagnosticsClient(processId)
    client.WriteDump(DumpType.Normal, model1 @@ "mem.dmp", logDumpGeneration=true)        

let pickTopSolutions() =
    let hiGains = t_log |> List.filter(fun x-> float x.Gain > 0.0) 
    let vecs = hiGains |> List.map toVec    
    let cfact xs k =  KMeansClustering.randomCentroids Probability.RNG.Value xs k |> List.map (fun (x:float[])->x,[])
    let cdist (x,_) y = KMeansClustering.euclidean x y
    let cavg (c,_) xs = (KMeansClustering.avgCentroid c xs),xs
    let centroids,_ = KMeansClustering.kmeans cdist cfact cavg vecs 5
    for c,_ in centroids do
        printfn "%A" c
    ()

let pickDistinctTopSolutions() =
    let hiGains = t_log |> List.filter(fun x-> float x.Gain > 0.0) 
    let vecs = hiGains |> List.map (fun x -> x.Gain, toVec x) |> List.sortByDescending fst
    let vecs = vecs |> List.distinctBy snd
    let vecs = List.truncate 10 vecs
    for (g,v) in vecs do 
        printfn $"// %0.5f{g}"
        printfn "%A" v
    ()
let showTopPickActionDists() =
    let hiGains = t_log |> List.filter(fun x-> float x.Gain > 0.0) 
    let vecs = hiGains |> List.map (fun x -> x.Gain, x) |> List.sortByDescending fst
    let vecs = vecs |> List.distinctBy snd

    let vecs = List.truncate 10 vecs
    for (g,v) in vecs do 
        printfn $"// %0.5f{g}"
        printfn "%A" v.ActDist
    ()

let plotHist2D() =
    hist2d "Gain vs Layers" (fun x -> float x.Gain) (fun x -> float x.Layers) t_log
    hist2d "Gain vs TrendWindowBars" (fun x -> float x.Gain) (fun x -> float x.TendWindowBars) t_log
    hist2d "Gain vs GoodBuyReward" (fun x -> float x.Gain) (fun x -> float x.GoodBuyInterReward) t_log
    hist2d "Gain vs GoodSellReward" (fun x -> float x.Gain) (fun x -> float x.GoodSellInterReward) t_log
    hist2d "Gain vs BadBuyIntrPenalty" (fun x -> float x.Gain) (fun x -> float x.BadBuyInterPenalty) t_log
    hist2d "Gain vs BadSellIntrPenalty" (fun x -> float x.Gain) (fun x -> float x.BadSellInterPenalty) t_log

let plotGainVsX() = 
    scatter "Gain vs Layers" (fun x -> float x.Gain) (fun x -> float x.Layers) t_log
    scatter "Gain vs TrendWindowBars" (fun x -> float x.Gain) (fun x -> float x.TendWindowBars) t_log
    scatter "Gain vs GoodBuyReward" (fun x -> float x.Gain) (fun x -> float x.GoodBuyInterReward) t_log
    scatter "Gain vs GoodSellReward" (fun x -> float x.Gain) (fun x -> float x.GoodSellInterReward) t_log
    scatter "Gain vs BadBuyIntrPenalty" (fun x -> float x.Gain) (fun x -> float x.BadBuyInterPenalty) t_log
    scatter "Gain vs BadSellIntrPenalty" (fun x -> float x.Gain) (fun x -> float x.BadSellInterPenalty) t_log
    scatter "Gain vs ImpossibleBuyPenalty" (fun x -> float x.Gain) (fun x -> float x.ImpossibleBuyPenalty) t_log
    scatter "Gain vs ImpossibleSellPenalty" (fun x -> float x.Gain) (fun x -> float x.ImpossibleSellPenalty) t_log
    scatter "Gain vs NonInvestmentPenalty" (fun x -> float x.Gain) (fun x -> float x.NonInvestmentPenalty) t_log

(*
pickTopSolutions()
dumpMemory()
pickDistinctTopSolutions()
showTopPickActionDists()
*)



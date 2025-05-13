#load "../scripts/packages.fsx"
open System
open System.IO
open Plotly.NET
open FSharp.Data
open System.Text.RegularExpressions

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
let t_log = T_Log.Load(logFilePath).Rows |> Seq.toList

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


// hist2d "Gain vs Layers" (fun x -> float x.Gain) (fun x -> float x.Layers) t_log
// hist2d "Gain vs TrendWindowBars" (fun x -> float x.Gain) (fun x -> float x.TendWindowBars) t_log
// hist2d "Gain vs GoodBuyReward" (fun x -> float x.Gain) (fun x -> float x.GoodBuyInterReward) t_log
// hist2d "Gain vs GoodSellReward" (fun x -> float x.Gain) (fun x -> float x.GoodSellInterReward) t_log
// hist2d "Gain vs BadBuyIntrPenalty" (fun x -> float x.Gain) (fun x -> float x.BadBuyInterPenalty) t_log
// hist2d "Gain vs BadSellIntrPenalty" (fun x -> float x.Gain) (fun x -> float x.BadSellInterPenalty) t_log

scatter "Gain vs Layers" (fun x -> float x.Gain) (fun x -> float x.Layers) t_log
scatter "Gain vs TrendWindowBars" (fun x -> float x.Gain) (fun x -> float x.TendWindowBars) t_log
scatter "Gain vs GoodBuyReward" (fun x -> float x.Gain) (fun x -> float x.GoodBuyInterReward) t_log
scatter "Gain vs GoodSellReward" (fun x -> float x.Gain) (fun x -> float x.GoodSellInterReward) t_log
scatter "Gain vs BadBuyIntrPenalty" (fun x -> float x.Gain) (fun x -> float x.BadBuyInterPenalty) t_log
scatter "Gain vs BadSellIntrPenalty" (fun x -> float x.Gain) (fun x -> float x.BadSellInterPenalty) t_log
scatter "Gain vs ImpossibleBuyPenalty" (fun x -> float x.Gain) (fun x -> float x.ImpossibleBuyPenalty) t_log
scatter "Gain vs ImpossibleSellPenalty" (fun x -> float x.Gain) (fun x -> float x.ImpossibleSellPenalty) t_log
scatter "Gain vs NonInvestmentPenalty" (fun x -> float x.Gain) (fun x -> float x.NonInvestmentPenalty) t_log



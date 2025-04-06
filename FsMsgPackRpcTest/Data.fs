module Data
open System
open System.IO
open MathNet.Numerics
open TsData
open Types

let avgPrice bar = 0.5 * (bar.High + bar.Low)        

let isNaN (c:float) = Double.IsNaN c || Double.IsInfinity c

let clipSlope (x:float) = 
    tanh (x/5.0)
    //max -5.0 (min 5.0 x) //clip slope to [-5,5]

let loadData() = 
    let data =
        File.ReadLines INPUT_FILE
        |> Seq.filter (fun l -> String.IsNullOrWhiteSpace l |> not)
        |> Seq.map(fun l -> 
            let xs = l.Split(',')
            let d =
                {
                    Time = DateTime.Parse xs.[1]
                    Open = float xs.[2]
                    High = float xs.[3]
                    Low = float xs.[4]
                    Close = float xs.[5]
                    Volume = float xs.[6]
                }
            d)
        |> Seq.toList
    let pd = data |> List.windowed TREND_WINDOW_BARS //|> List.truncate (100000 * 4)
    let pds =
        pd
        |> List.mapi (fun i xs ->
            let x = List.last xs
            let y = xs.[xs.Length - 2]
            let pts1 = xs |> List.map avgPrice
            let ys1N = LinearAlgebra.Double.Vector.Build.DenseOfEnumerable(pts1).Normalize(1.0)
            let xs1N = LinearAlgebra.Double.Vector.Build.DenseOfEnumerable([0 .. pts1.Length]).Normalize(1.0)
            let pts2 = xs |> List.skip (xs.Length/2) |> List.map avgPrice
            let ys2N = LinearAlgebra.Double.Vector.Build.DenseOfEnumerable(pts2).Normalize(1.0)
            let xs2N = LinearAlgebra.Double.Vector.Build.DenseOfEnumerable([0 .. pts2.Length]).Normalize(1.0)
            let struct(_,s1) = LinearRegression.SimpleRegression.Fit(Seq.zip xs1N ys1N) 
            let struct(_,s2) = LinearRegression.SimpleRegression.Fit(Seq.zip xs2N ys2N)
            let cs1 = clipSlope s1
            let cs2 = clipSlope s2
            let d =
                {
                    TrendLong = cs1
                    TrendShort = cs2

                    NOpen = (y.Open - x.Open) / x.Open 
                    NHigh = (y.High - x.High) / x.High 
                    NLow =  (y.Low - x.Low) / x.Low 
                    NClose = (y.Close - x.Close) / x.Close 
                    NVolume = (y.Volume - x.Volume) / x.Close

                    //NOpen = exp(y.Open  / x.Open) 
                    //NHigh = exp(y.High /  x.High) 
                    //NLow =  exp(y.Low / x.Low)   
                    //NClose = exp(y.Close / x.Close)
                    //NVolume = exp(y.Volume / x.Volume) 

                    //TrendLong = cs1
                    //TrendShort = cs2
                    //NOpen = log(y.Open/x.Open) |> max -18. //- 1.0
                    //NHigh = log(y.High/x.High) |> max -18. //- 1.0
                    //NLow =  log(y.Low/x.Low)   |> max -18. //- 1.0
                    //NClose = log(y.Close/x.Close) |> max -18.// - 1.0
                    //NVolume = log(y.Volume/x.Volume) |> max -18. //- 1.0
                    //NOpen = (y.Open/x.Open) //|> max -18. //- 1.0
                    //NHigh = (y.High/x.High) //|> max -18. //- 1.0
                    //NLow =  (y.Low/x.Low)   //|> max -18. //- 1.0
                    //NClose = (y.Close/x.Close) //|> max -18.// - 1.0
                    //NVolume = (y.Volume/x.Volume) //|> max -18. //- 1.0
                    Bar  = y
                }
            if isNaN d.NOpen ||isNaN d.NHigh || isNaN d.NLow || isNaN d.NClose || isNaN d.NVolume then
                failwith "nan in data"
            (x,y),d
        )
    let xl = pds |> List.last
    pds |> List.map snd

let dataRaw = loadData() //|> List.truncate (EPISODE_LENGTH * 10)
let TRAIN_SIZE = float dataRaw.Length * TRAIN_FRAC |> int
let dataTrain = dataRaw |> Seq.truncate TRAIN_SIZE |> Seq.toArray
let dataTest = dataRaw |> Seq.skip TRAIN_SIZE |> Seq.toArray
let pricesTrain = {prices = dataTrain}
let pricesTest = {prices = dataTest}
    
let resetLogs() =
    let logDir = root @@ "logs"
    if Directory.Exists logDir |> not then 
        Directory.CreateDirectory logDir |> ignore
    else
        Directory.GetFiles(logDir) |> Seq.iter File.Delete

let logger = MailboxProcessor.Start(fun inbox -> 
    async {
        while true do
            let! (agentId:int,parmsId:int,line:string) = inbox.Receive()
            try
                let fn = root @@ "logs" @@ $"log_{agentId}_{parmsId}.csv"
                if File.Exists fn |> not then
                    //let logLine = $"{s.AgentId},{s.Episode},{s.Step.Num},{action},{avgP},{s.CashOnHand},{s.Stock},{reward},{sGain},{parms.RunId}"
                    let header = "agentId,episode,step,action,price,cash,stock,reward,gain,parmId"
                    File.AppendAllLines(fn,[header;line])
                else
                    File.AppendAllLines(fn,[line])
            with ex -> 
                printfn $"logger: {ex.Message}"
    })



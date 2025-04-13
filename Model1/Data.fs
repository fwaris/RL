module Data
open System
open System.IO
open MathNet.Numerics
open TsData
open Types
open System.Numerics
open TorchSharp
open System.Runtime.Intrinsics.Arm
open TorchSharp
open MathNet.Numerics
open Plotly.NET

let avgPrice bar = 0.5 * (bar.High + bar.Low)        

let isNaN (c:float) = Double.IsNaN c || Double.IsInfinity c

let clipSlope (x:float) = 
    tanh (x/5.0)
    //max -5.0 (min 5.0 x) //clip slope to [-5,5]

let getSlope (pts:float list) =
    let ys = LinearAlgebra.Double.Vector.Build.DenseOfEnumerable(pts).Normalize(1.0)
    let xs = LinearAlgebra.Double.Vector.Build.DenseOfEnumerable([0 .. pts.Length]).Normalize(1.0)
    let struct(_,slope) = LinearRegression.SimpleRegression.Fit(Seq.zip xs ys) 
    clipSlope slope

let ftTrans (pts:float list) =
    let pts = pts |> List.map (fun x -> Complex(x,0)) |> List.toArray
    MathNet.Numerics.IntegralTransforms.Fourier.Forward(pts)
    pts

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
        |> Seq.filter (fun x -> x.High > 0. && x.Low > 0. && x.Open > 0. && x.Close > 0.)
        |> Seq.toList

    let pd = data |> List.windowed TREND_WINDOW_BARS //|> List.truncate (100000 * 4)
    let pds =
        pd        
        |> List.mapi (fun i xs ->
            let x = List.last xs
            let y = xs.[xs.Length - 2]
            let pts = xs |> List.map avgPrice
            let priceReturns = pts |> List.pairwise |> List.map (fun (a,b) -> (b-a) / a)
            let nrets = LinearAlgebra.CreateVector.DenseOfEnumerable(priceReturns).Normalize(1.0)
            use d_pts = torch.tensor(nrets.ToArray(), dtype=torch.float)            
            use ptsFFt = torch.fft.rfft(d_pts, norm=FFTNormType.Forward)
            use ptsFFtR = ptsFFt.real
            let t_ptsFFT = Fun.Tensor.getData<float32>(ptsFFtR)
            let ptsMed = pts |> List.skip (xs.Length / 3 * 1)
            let ptsShort = pts |> List.skip (xs.Length / 3 * 2 )
            let slope = getSlope pts
            let slopeMed = getSlope ptsMed
            let slopeShort = getSlope ptsShort
            let d =
                {
                    Freq1 = float t_ptsFFT.[0]
                    Freq2 = float t_ptsFFT.[1]
                    TrendLong = slope
                    TrendMed = slopeMed
                    TrendShort = slopeShort

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

let dataRaw = loadData() |> List.skip (EPISODE_LENGTH * 10) |> List.take (EPISODE_LENGTH * 7)
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
            let! (episode:int,parmsId:int,line:string) = inbox.Receive()
            try
                let fn = root @@ "logs" @@ $"log_{episode}_{parmsId}.csv"
                if File.Exists fn |> not then
                    //let logLine = $"{s.AgentId},{s.Episode},{s.Step.Num},{action},{avgP},{s.CashOnHand},{s.Stock},{reward},{sGain},{parms.RunId}"
                    let header = "agentId,episode,step,action,price,cash,stock,reward,gain,parmId,market,isDone"
                    File.AppendAllLines(fn,[header;line])
                else
                    File.AppendAllLines(fn,[line])
            with ex -> 
                printfn $"logger: {ex.Message}"
    })


let trainMarkets =
    let episodes = dataTrain.Length / EPISODE_LENGTH    
    let idxs = [0 .. episodes-1] |> List.map (fun i -> i * EPISODE_LENGTH)
    idxs
    |> List.map(fun i -> 
        let endIdx = i + EPISODE_LENGTH - 1
        if endIdx <= i then failwith $"Invalid index {i}"
        {Market = pricesTrain; StartIndex=i; EndIndex=endIdx})
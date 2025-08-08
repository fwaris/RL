module Data
open System
open System.IO
open MathNet.Numerics
open TsData
open Types
open System.Numerics
open TorchSharp
open TorchSharp
open MathNet.Numerics.LinearAlgebra
open Plotly.NET
open Plotly.NET.StyleParam

let effectivePrice bar =  bar.Close// 0.5 * (bar.High + bar.Low)        

let isNaN (c:float) = Double.IsNaN c || Double.IsInfinity c

let clipSlope (x:float) = 
    tanh (x/5.0)
    //max -5.0 (min 5.0 x) //clip slope to [-5,5]

let inline scaleN vs =
    let vs = Seq.map float vs
    let vmin = Seq.min vs 
    let vmax = Seq.max vs
    let range = vmax - vmin
    vs |> Seq.map(fun x -> (x - vmin) / range)

let getSlope (pts:float list) =    
    let spts = scaleN pts
    let ys = LinearAlgebra.Double.Vector.Build.DenseOfEnumerable(spts)
    let xs = LinearAlgebra.Double.Vector.Build.DenseOfEnumerable([1 .. pts.Length] |> scaleN)
    let struct(_,slope) = LinearRegression.SimpleRegression.Fit(Seq.zip xs ys) 
    clipSlope slope

let ftTrans (pts:float list) =
    let pts = pts |> List.map (fun x -> Complex(x,0)) |> List.toArray
    MathNet.Numerics.IntegralTransforms.Fourier.Forward(pts)
    pts

let private loadBaseData() =
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

let private loadData tp = 
    let data = loadBaseData()
    let pd (tp:TuneParms) = data |> List.windowed tp.TrendWindowBars 
    let pds tp =
        pd tp   
        |> List.mapi (fun i xs ->
            let currBar = List.last xs
            let prevBar = xs.[xs.Length - 2]
            let pctRange = xs |> List.map (fun b -> (b.High - b.Close) / b.Close |> max -1.0 |> min 1.0)
            let vols = xs |> List.map (fun b -> b.Volume)
            let pctRangeVol = xs |> List.map (fun b -> (b.High - b.Close) / b.Close / b.Volume)
            let avgPrices = xs |> List.map effectivePrice
            //let priceReturns = avgPrices |> List.pairwise |> List.map (fun (a,b) -> (b-a) / a)             
            //let scaledReturns = scaleN priceReturns |> Seq.toList
            //let nrets = LinearAlgebra.CreateVector.DenseOfEnumerable(scaledReturns)
            //use d_pts = torch.tensor(nrets.ToArray(), dtype=torch.float)
            //use ptsFFt = torch.fft.rfft(d_pts, norm=FFTNormType.Forward)
            //use ptsFFtR = ptsFFt.real
            //let t_ptsFFT = Fun.Tensor.getData<float32>(ptsFFtR)
            let avgPricesMed = avgPrices |> List.skip (xs.Length / 3 * 1)
            let avgPricesShort = avgPrices |> List.skip (xs.Length / 3 * 2 )
            let slope = getSlope avgPrices
            let slopeMed = getSlope avgPricesMed
            let slopeShort = getSlope avgPricesShort

            let statsRange = pctRange |> Statistics.DescriptiveStatistics 
            let statsVol = vols |> Statistics.DescriptiveStatistics
            let d =
                {
                    Freq1 = float statsRange.Kurtosis
                    Freq2 = float statsVol.Kurtosis
                    TrendLong = slope
                    TrendMed = slopeMed
                    TrendShort = slopeShort

                    NOpen = (currBar.Open - prevBar.Open)/ prevBar.Open 
                    NHigh = (currBar.High - prevBar.High) / prevBar.High 
                    NLow =  (currBar.Low - prevBar.Low) / prevBar.Low 
                    NClose = (currBar.Close - prevBar.Close) / prevBar.Close 
                    NVolume = (prevBar.Volume - currBar.Volume) / prevBar.Volume 

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
                    Bar  = prevBar
                }
            if isNaN d.NOpen ||isNaN d.NHigh || isNaN d.NLow || isNaN d.NClose || isNaN d.NVolume then
                failwith "nan in data"
            (currBar,prevBar),d
        )    
    pds tp |> List.map snd

let numMarketSlices (xs:_[]) = xs.Length / EPISODE_LENGTH

let getSlices tp = 
    let data = loadBaseData()
    let dataSet = data |> List.windowed tp.TrendWindowBars 
    let trainSize  = float dataSet.Length * TRAIN_FRAC |> int
    dataSet |> Seq.truncate trainSize |> Seq.toArray    
    
let testTrain tp = 
    let dataSet = loadData tp
    let dataSet = tp.SkipBars |> Option.map (fun skip -> dataSet |> List.skip skip) |> Option.defaultValue dataSet
    let dataSet = tp.TakeBars |> Option.map (fun take -> dataSet |> List.take take) |> Option.defaultValue dataSet    
    let trainSize  = float dataSet.Length * TRAIN_FRAC |> int
    let dataTrain = dataSet |> Seq.truncate trainSize |> Seq.toArray
    let dataTest = dataSet |> Seq.skip trainSize |> Seq.toArray
    dataTrain, dataTest
let resetLogs() =
    let logDir = root @@ "logs"
    if Directory.Exists logDir |> not then 
        Directory.CreateDirectory logDir |> ignore
    else
        Directory.GetFiles(logDir) |> Seq.iter File.Delete

let logger = MailboxProcessor.Start(fun inbox -> 
    async {
        while true do
            let! (episode:int,parmsId:string,line:string) = inbox.Receive()
            try
                let fn = root @@ "logs" @@ $"log_{episode}_{parmsId}.csv"
                if File.Exists fn |> not then
                    Types.ensureDirForFilePath fn
                    //let logLine = $"{s.AgentId},{s.Episode},{s.Step.Num},{action},{avgP},{s.CashOnHand},{s.Stock},{reward},{sGain},{parms.RunId}"
                    let header = "Timestamp,agentId,episode,step,action,price,cash,stock,reward,gain,parmId,market,isDone"
                    File.AppendAllLines(fn,[header;line])
                else
                    File.AppendAllLines(fn,[line])
            with ex -> 
                printfn $"logger: {ex.Message}"
    })


let episodeLengthMarketSlices (trainData:NBar[]) =
    let episodes = trainData.Length / EPISODE_LENGTH    
    let idxs = [0 .. episodes-1] |> List.map (fun i -> i * EPISODE_LENGTH)
    idxs
    |> List.map(fun i -> 
        let endIdx = i + EPISODE_LENGTH - 1
        if endIdx <= i then failwith $"Invalid index {i}"
        {Market = {prices=trainData}; StartIndex=i; EndIndex=endIdx})

let numberOfOEpisodeLengthMarketSlices<'a> (trainData:'a[]) =
    let episodes = trainData.Length / EPISODE_LENGTH    
    let idxs = [0 .. episodes-1] |> List.map (fun i -> i * EPISODE_LENGTH)
    idxs
    |> List.map(fun i -> 
        let endIdx = i + EPISODE_LENGTH - 1
        if endIdx <= i then failwith $"Invalid index {i}"
        {|StartIndex=i; EndIndex=endIdx|})
    |> List.length


let singleMarketSlice (bars:NBar[]) = 
    {Market = {prices = bars}; StartIndex=0; EndIndex=bars.Length-1}

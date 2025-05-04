#r "nuget: MathNet.Numerics"
#r "nuget: FSharp.Data"
open MathNet.Numerics
open FSharp.Data
open MathNet.Numerics.Statistics

let vals : float list = [87029; 113407; 84843; 104994;  99327;  92052; 60684]
let mean = List.average vals
let sigma = (vals |> List.map(fun x -> (x - mean)**2.0) |> List.sum )/ (float vals.Length - 1.0) |> sqrt
let se = sigma / sqrt (float vals.Length)
let z = MathNet.Numerics.Distributions.Normal.InvCDF(0.0,1.0,0.025)
let ``z * se`` = z * se

let lo,hi = mean - ``z * se`` , mean + ``z * se``

let m1  = Statistics.Statistics.Mean(vals)
let s1 = Statistics.Statistics.StandardDeviation(vals)


type TExp = CsvProvider< @"C:\Users\Faisa\Downloads\L3_ Empirical Variance Quiz Data - Sheet1.csv">

let texs=TExp.GetSample()
let rows = texs.Rows |> Seq.toList
let dvals = rows |> List.map (fun x -> float x.Diff)

let dMean = Statistics.Statistics.Mean(dvals)
let dSigma = Statistics.Statistics.StandardDeviation(dvals)
let z95 = Distributions.Normal.InvCDF(0.0,1.0,0.025) |> abs
let dSE = dSigma * z95
let cL,cH = dMean - (dSigma * z95), dMean + (dSigma * z95)

let n = float dvals.Length
let n25 = n * 0.025  |> int
let n95 = n * 0.95 |> int

let dVal95 = 
    dvals
    |> List.sort 
    |> List.skip n25
    |> List.take n95

let eDMin, dDMax = List.min dVal95, List.max dVal95

let binDist = Distributions.Binomial(0.5,7)
let oneTailP = binDist.Probability(7)
let twoTailP = oneTailP * 2.0











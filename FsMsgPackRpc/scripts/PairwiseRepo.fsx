open System
open System.IO
type Bar =
    {
        Open  : float 
        High  : float
        Low   : float
        Close : float
        Volume : float
        Time : DateTime        
    }
type NBar =
    {
        NOpen  : float 
        NHigh  : float
        NLow   : float
        NClose : float
        NVolume : float
        Bar     : Bar
    }

let fn = @"E:\s\test_data.csv"
let isNaN (c:float) = Double.IsNaN c || Double.IsInfinity c

let loadData() = 
    let data =
        File.ReadLines fn
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
    let pd = data |> List.pairwise |> List.truncate 40001 //<== reducing this 10K or less works
    let pds =
        ([],pd)
        ||> List.fold (fun acc (x,y) ->
            let d =
                {
                    NOpen = (y.Open/x.Open) - 1.0
                    NHigh = (y.High/x.High) - 1.0
                    NLow =  (y.Low/x.Low)   - 1.0
                    NClose = (y.Close/x.Close)  - 1.0
                    NVolume = (y.Volume/x.Volume) - 1.0
                    Bar  = y
                }
            if isNaN d.NOpen ||isNaN d.NHigh || isNaN d.NLow || isNaN d.NClose || isNaN d.NVolume then
                failwith "nan in data"
            ((x,y),d)::acc
        )
    let xl = pds |> List.last
    pds |> List.map snd

let data = loadData()
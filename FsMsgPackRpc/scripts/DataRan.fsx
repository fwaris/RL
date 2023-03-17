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

let fn = @"E:\s\tradestation\mes_hist_td.csv"
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
    data


let changeData (data: Bar list) = 
    let rng = System.Random()
    data
    |> List.map(fun b ->
        { b with
              Open = b.Open * rng.NextDouble()
              High = b.High * rng.NextDouble()
              Low  = b.Low * rng.NextDouble()
              Close = b.Close * rng.NextDouble()
              Volume = b.Volume * rng.NextDouble()
        }
    )

let saveData (data: Bar list) =
    let fn = @"E:\s\test_data.csv"
    let lines = data |> Seq.map(fun b -> String.Join(",", "S", b.Time, b.Open, b.High, b.Low, b.Close, b.Volume))
    File.WriteAllLines(fn,lines)

let data = loadData()
data |> changeData |> saveData

module TsData 
open System
open MBrace.FsPickler

type Bar =
    {
        Open  : float 
        High  : float
        Low   : float
        Close : float
        Volume : float
        Time : DateTime        
    }

//let saveBars (file:string) (bars:TsWeb.TsApi.TsClient.Bar[]) =
//    let bars2 = 
//        bars
//        |> Array.map(fun x->
//        (
//            x.Open |> float,
//            x.High |> float,
//            x.Low  |> float,
//            x.Close |> float,
//            x.TotalVolume |> float,
//            x.TimeStamp |> string |> DateTime.Parse
//        ))
//    let ser = FsPickler.CreateBinarySerializer()
//    use strw = System.IO.File.Create(file)
//    ser.Serialize(strw,bars2)

let loadBars (file:string) = 
    let ser = FsPickler.CreateBinarySerializer()
    let str = System.IO.File.OpenRead file
    let raw = ser.Deserialize<(float*float*float*float*float*DateTime)[]>(str)
    raw 
    |> Array.map(fun (o,h,l,c,v,d) ->
        {
            Open  = o
            High  = h
            Low   = l
            Close = c
            Volume = v
            Time = d
        })

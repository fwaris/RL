#r "nuget: AVLoop"

open FSharp.Compiler.Interactive
open AVLoop


let install(theme) =
        fsi.EventLoop <- {new IEventLoop with 
                                member x.Run() = 
                                    createApp(theme, [||])
                                    false //dummy
                                member x.Invoke(f) = disp f
                                member x.ScheduleRestart() = () //dummy
                        }

install(Default,Dark) //wait till initialization message before submitting more code

open Avalonia
open Avalonia.Media.Imaging
open System.IO

let showImage (bytes:byte[]) =    
    let win1 = Controls.Window()
    win1.Width <- 300
    win1.Height <- 300
    let img = Controls.Image()
    use ms = new MemoryStream(bytes)
    img.Source <- new Bitmap(ms)
    win1.Content <- img
    win1.Show()


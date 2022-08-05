#load "packages.fsx"

let go = ref true
CarEnvironment.startRandomAgent go |> Async.Start  // start agent

go.Value <- false //stop agent


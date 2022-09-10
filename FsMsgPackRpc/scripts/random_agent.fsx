#load "packages.fsx"

let go = ref true
let doLog = ref true
CarEnvironment.startRandomAgent doLog go |> Async.Start  // start agent

go.Value <- false //stop agent


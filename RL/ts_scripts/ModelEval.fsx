#load "../scripts/packages.fsx"
#load "../TsData.fs"
#load "../RL.fs"
#I @"../../Model1"
#load "Types.fs"
#load "Data.fs"
#load "Agent.fs"
#load "Policy.fs"
#load "Test.fs"
#load "Train.fs"
#load "Model.fs"

Test.evalModels Model.parms.Head

open System
open MessagePack
open MessagePack.FSharp
open MessagePack.Resolvers
open FsMsgPackRpc

let address = "localhost"
let port = 41451
let resolver = Resolvers.CompositeResolver.Create(
                FSharpResolver.Instance,
                StandardResolver.Instance)
let options = MessagePackSerializerOptions.Standard.WithResolver(resolver)


type AimC(options) =
    inherit Client(options) 

    member this.getServerVersion() = 
        let name = nameof this.getServerVersion
        base.Call<_,int>(name,[||]) 
        

let c1 = new AimC(options)
c1.Connect(address,port)
let ts = c1.getServerVersion() |> Async.AwaitTask |> Async.RunSynchronously

(c1 :> IDisposable).Dispose()



namespace FsMsgPackRpc
open System
open System.IO
open System.Buffers
open MessagePack
open System.Net.Sockets
open System.Threading

type ServerResp = Data of obj | Error of obj
type Msg = Req of (int*(Type*AsyncReplyChannel<ServerResp>)) | Resp of int*obj*byte[]

module Matcher = 
    ///agent to match responses to pending requests
    ///Note: MsgPack rpc allows for out-of-order responses 
    let createAgent options (cts:CancellationTokenSource) =
            (fun (inbox:MailboxProcessor<Msg>) ->
                let pending = new System.Collections.Generic.Dictionary<int,(Type*AsyncReplyChannel<ServerResp>)>()
                async {
                    while not cts.IsCancellationRequested do
                        try                        
                            match! inbox.Receive() with
                            | Req (id,rc) -> pending.Add(id,rc)                        
                            | Resp (id,err,resp) ->                                
                                match pending.TryGetValue id with
                                | true,(t,rc) -> 
                                    pending.Remove(id) |> ignore
                                    if err <> null then
                                        rc.Reply(Error err)
                                    else
                                        use ms = new MemoryStream(resp)
                                        let! obj = task {return! MessagePackSerializer.DeserializeAsync(t,ms,options=options)} |> Async.AwaitTask                                    
                                        rc.Reply(Data obj)                                   
                                | _ -> printfn $"unmatched response for id {id}"
                        with ex -> 
                            printfn "%A" ex.Message
            })

///threadsafe client for interacting with MessagePack RPC servers (https://github.com/msgpack-rpc/msgpack-rpc)
type Client(options:MessagePack.MessagePackSerializerOptions) =    
    let cts = new CancellationTokenSource()
    let tcpClient = new TcpClient()    
    let mutable id = -1
    let nextId() = Interlocked.Increment(&id)
    let mbp = lazy(MailboxProcessor.Start(Matcher.createAgent options cts,cts.Token))
    let sem = new ManualResetEvent(true)

    //message receive loop
    let receive() =
        task {
            try
                use strm = new MessagePackStreamReader(tcpClient.GetStream(),leaveOpen=true)
                while not cts.IsCancellationRequested do
                    let! h = strm.ReadArrayHeaderAsync(cts.Token)
                    let! m = strm.ReadAsync(cts.Token)
                    let! id = strm.ReadAsync(cts.Token)
                    let! err = strm.ReadAsync(cts.Token)
                    let! data = strm.ReadAsync(cts.Token)
                    if h > 0 && m.HasValue && id.HasValue && err.HasValue && data.HasValue then                        
                        let mutable idV = id.Value
                        let msgId = MessagePackSerializer.Deserialize<int>(&idV)                        
                        let mutable errV = err.Value
                        let errObj = MessagePackSerializer.Deserialize<obj>(&errV)
                        let bytes = data.Value.ToArray()
                        let resp = Resp(msgId,errObj,bytes)
                        printfn "%A" resp
                        mbp.Value.Post resp
            with ex -> 
                printfn "%A" ex.Message
        }        

    member _.Connect (address:string,port:int) =  
        tcpClient.Connect(address,port)        
        receive() |> ignore

    member private _.Disconnect() =
        cts.CancelAfter(100)
        tcpClient.Dispose()
        sem.Dispose()

    member _.TcpClient = tcpClient
    
    ///sends a message and waits for a response
    member _.Call<'req,'resp> (method:string,req:'req) =
        let msgId = nextId()
        let reqMsg:obj[] = [|0; msgId; method; req |]        
        task {        
            let! resp = mbp.Value.PostAndTryAsyncReply((fun rc -> Req (msgId, (typeof<'resp>, rc))),timeout=5000) |> Async.StartChild
            let! _ = Async.AwaitWaitHandle sem
            let _ = sem.Reset()
            try
                do! MessagePackSerializer.SerializeAsync(tcpClient.GetStream(),reqMsg)
            finally
                sem.Set() |> ignore
            match! resp with 
            | Some (Data o) -> return (o :?> 'resp)
            | Some (Error e) -> return failwith $"%A{e}"
            | None -> return failwith "timeout"
        }

    ///sends a one-way notification (i.e. response not expected)
    member _.Notify<'req> (method:string,req:'req) =
        let msgId = nextId()
        let reqMsg:obj[] = [|0; msgId; method; req |]        
        task {        
            let! _ = Async.AwaitWaitHandle sem
            let _ = sem.Reset()
            try
                do! MessagePackSerializer.SerializeAsync(tcpClient.GetStream(),reqMsg)
            finally
                sem.Set() |> ignore
        }
        
    interface IDisposable with
        member this.Dispose() = this.Disconnect()
                


        
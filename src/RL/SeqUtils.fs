module SeqUtils
open System
open TorchSharp
open TorchSharp.Fun
open type torch.TensorIndex
let ``...`` = Ellipsis
let ``:`` = Colon
let IDX_NULL = Nullable<int64>()
let IDX_ZERO = torch.TensorIndex.Slice(0L)
let FIRST = torch.TensorIndex.Single(0L)
let LAST = torch.TensorIndex.Single(-1L)

module PositionalEncoder =
    let create dropout dmodel maxLen =
        let dropout = torch.nn.Dropout(dropout) |> M
        let posTnsr = torch.zeros([|maxLen; dmodel|])       //5000 x 200
        let position = torch.arange(maxLen).unsqueeze(1L)   //5000 x 1
        let divTerm1 = -(10000.0.ToTensor().log()) / ((float dmodel).ToScalar())
        let divTerm2 = torch.arange(0L,dmodel,2L)               //100
        let divTerm3 = (divTerm2 * divTerm1.ToScalar())
        let divTerm = divTerm3.exp()   //100
        //let divTermT = Tensor.getDataNested<float32> divTerm
        posTnsr.[ ``:``, Slice(0L,IDX_NULL,2L) ] <- (position * divTerm).sin()
        posTnsr.[ ``:``, Slice(1L,IDX_NULL,2L) ] <- (position * divTerm).cos()
        let pe = posTnsr.unsqueeze(0L).transpose(0L,1L)
        pe.name <- "pe"
        let peRef = ref pe

        let mdl = 
            F [] [dropout; peRef] (fun t -> 
                let pos = peRef.Value[Slice(IDX_NULL,t.shape.[0]), Slice()]
                use x = t + pos
                dropout.forward(x))
        mdl

module Masks =
    open TorchSharp
    open System
    let generateSubsequentMask size (device:TorchSharp.torch.Device) =
        let mask = torch.ones([|size; size|]).tril()
        let subseqMask=torch.zeros(size,size).masked_fill(mask.eq(0.f.ToScalar()), (-infinityf).ToScalar())
        subseqMask.``to``(device)

module Seq =
    let stridedChunks stride chunkSize (source : seq<_> ) =
        if stride <= 0 then failwith "stride should be a positive integer"
        if chunkSize <= 0 then failwith "chunkSize should be a positive integer"
        if stride > chunkSize then failwith "stride > chunkSize not supported"
        seq {
                use e = source.GetEnumerator()
                let buffer = Array.zeroCreate chunkSize
                let go = ref true
                let i = ref 0
                let fillBuffer() =
                    while i.Value < chunkSize && e.MoveNext() do
                        buffer.[i.Value] <- e.Current
                        i.Value <- i.Value + 1
                let copyAndSlideBack() =
                    let res = Array.copy buffer
                    System.Array.Copy(res,stride,buffer,0,chunkSize-stride)
                    res
                while go.Value do
                    fillBuffer()
                    let res = copyAndSlideBack()
                    go.Value <- i.Value = chunkSize
                    if go.Value then
                        i.Value <- chunkSize-stride
                        yield res
                    else
                        let m = res.[0..i.Value-1]
                        if m.Length > 0 then
                            yield m
                    }

    (*
    let t = [0;1;2;3;4;5;6]
    stridedChunks 2 4 t  // val it : seq<int []> = seq [[|0; 1; 2; 3|]; [|2; 3; 4; 5|]; [|4; 5; 6|]]
    stridedChunks 1 3 t |> Seq.toArray // val it : int [] [] = [|[|0; 1; 2|]; [|1; 2; 3|]; [|2; 3; 4|]; [|3; 4; 5|]; [|4; 5; 6|]; [|5; 6|]|]
    stridedChunks 1 1 t |> Seq.toArray //val it : int [] [] = [|[|0|]; [|1|]; [|2|]; [|3|]; [|4|]; [|5|]; [|6|]|]
    *)
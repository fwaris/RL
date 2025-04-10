module Model
open System
open TorchSharp
open TorchSharp.Fun
open DQN
open SeqUtils
open Types

let NUM_MKT_SLICES = Data.TRAIN_SIZE / EPISODE_LENGTH

let trainMarkets =
    let episodes = Data.dataTrain.Length / EPISODE_LENGTH    
    let idxs = [0 .. episodes-1] |> List.map (fun i -> i * EPISODE_LENGTH)
    idxs
    |> List.map(fun i -> 
        let endIdx = i + EPISODE_LENGTH - 1
        if endIdx <= i then failwith $"Invalid index {i}"
        {Market = Data.pricesTrain; StartIndex=i; EndIndex=endIdx})


let parms1 id (lr,layers)  = 
    let emsize = 64
    let dropout = 0.1
    let max_seq = LOOKBACK
    let nheads = 1
    let nlayers = layers

    let createModel() = 
        let proj = torch.nn.Linear(INPUT_DIM,emsize)
        let pos_emb = torch.nn.EmbeddingBag(LOOKBACK,emsize)
        let encoder_layer = torch.nn.TransformerEncoderLayer(emsize,nheads,emsize,dropout)
        let transformer_encoder = torch.nn.TransformerEncoder(encoder_layer,nlayers)        
        let sqrtEmbSz = (sqrt (float emsize)).ToScalar()
        let projOut = torch.nn.Linear(emsize,ACTIONS)
        //et activation = torch.nn.Tanh()
        let initRange = 0.1
        let mdl = 
            F [] [proj; pos_emb; transformer_encoder; projOut]  (fun inp -> //B x S x 5
                use p = proj.forward(inp) // B x S x emsize                
                let batchSize::seqLen::_ = p.size() |> Seq.toList
                use pos = torch.arange(seqLen,device=inp.device).unsqueeze(1)
                use pos_emb = pos_emb.call(pos)
                use projWithPos = p + pos_emb
                use pB2 = projWithPos.permute(1,0,2) //batch second - S x B x emsize                
                use enc = transformer_encoder.call(pB2) //S x B x emsize
                use encB = enc.permute(1,0,2)  //batch first  // B x S x emsize
                use dec = encB.[``:``,LAST,``:``]    //keep last value as output to compare with target - B x emsize
                let pout = projOut.forward(dec) //B x ACTIONS
                //let act = activation.forward pout
                //let t_act = Tensor.getDataNested<float32> act
                pout
            )
        mdl
    let model = DQNModel.create createModel
    let exp = {Decay=0.9995; Min=0.01; WarupSteps=5000}
    let DQN = DQN.create model 0.99999f exp ACTIONS device
    {Parms.Default createModel DQN lr id with 
        SyncEverySteps = 10000
        BatchSize = 32
        Epochs = 1000}

let lrs = [0.001,3L]//; 0.001,8L; 0.001,10]///; 0.0001; 0.0002; 0.00001]
let parms = lrs |> List.mapi (fun i lr -> parms1 i lr)

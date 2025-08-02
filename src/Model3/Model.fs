module Model
open System
open TorchSharp
open TorchSharp.Fun
open DQN
open SeqUtils
open Types

let parms1 id (lr,tp:TuneParms) = 
    let emsize = 64
    let dropout = 0.1
    let nheads = 1
    let nlayers = tp.Layers

    let createModel() = 
        let proj = torch.nn.Linear(INPUT_DIM,emsize)
        let pos_emb = torch.nn.EmbeddingBag(tp.Lookback,emsize)
        let encoder_layer = torch.nn.TransformerEncoderLayer(emsize,nheads,emsize,dropout)
        let transformer_encoder = torch.nn.TransformerEncoder(encoder_layer,nlayers)                
        let projOut = torch.nn.Linear(emsize,ACTIONS)
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
    let exp = {Decay=0.9995; Min=0.01; WarupSteps=WARMUP}
    let DQN = DQN.create model 0.99999f exp ACTIONS
    {Parms.Default createModel DQN lr id with 
        SyncEverySteps = 3000
        BatchSize = 32
        Epochs = EPOCHS
        TuneParms = tp}

let tp = //0.17,-0.8300000000000001,-0.57,0.05,-0.66,0,0,9,80,26
         //GoodBuyInterReward,BadBuyInterPenalty,ImpossibleBuyPenalty,GoodSellInterReward,BadSellInterPenalty,ImpossibleSellPenalty,NonInvestmentPenalty,Layers,TendWindowBars,Lookback
    {
        GoodBuyInterReward = 0.17
        BadBuyInterPenalty = -0.83
        ImpossibleBuyPenalty = -0.057
        GoodSellInterReward = 0.05
        BadSellInterPenalty = -0.66
        ImpossibleSellPenalty = 0.0
        NonInvestmentPenalty = 0.0
        Layers = 9L
        Lookback = 26L // LOOKBACK
        TrendWindowBars = 80//TREND_WINDOW_BARS
    }   

let parmSpace = [0.001,tp]//; 0.001,8L; 0.001,10]///; 0.0001; 0.0002; 0.00001]
let parms = parmSpace |> List.mapi (fun i ps -> parms1 i ps)



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
        SyncEverySteps = 500
        BatchSize = 32
        Epochs = EPOCHS
        TuneParms = tp}

let tp = //0.34,-0.84,-0.57,0.98,-0.16,0,0,10
         //a.GoodBuyInterReward, a.BadBuyInterPenalty, a.ImpossibleBuyPenalty, a.GoodSellInterReward, a.BadSellInterPenalty, a.ImpossibleSellPenalty, a.NonInvestmentPenalty, a.Layers
    { TuneParms.Default with        
        GoodBuyInterReward = 0.34
        BadBuyInterPenalty = -0.84
        ImpossibleBuyPenalty = -0.057
        GoodSellInterReward = 0.98
        BadSellInterPenalty = -0.16
        ImpossibleSellPenalty = 0.0
        NonInvestmentPenalty = 0.0
        Layers = 10L
        Lookback = 30L // LOOKBACK
        TrendWindowBars = 60//TREND_WINDOW_BARS
    }   

let caSols = 
    [
        //(1.9252314606741572M,
        [|1.0; 3.0; 10.0; 0.6; 1.0; -0.09; -0.34; -0.86; -0.85; 0.0|]
        //(1.697865168539326M, 
        [|1.0; 3.0; 10.0; 0.72; 1.0; 0.0; -1.0; -1.0; -0.6; -0.08|]
        //(1.626997752808989M,
        [|1.0; 3.0; 10.0; 0.74; 0.98; -0.2; -0.71; -0.75; -0.92; 0.0|]
        //(1.5886112359550564M,
        [|3.0; 3.0; 10.0; 0.72; 0.83; -0.3; -0.63; -0.61; -0.6; -0.04|]
        //(1.5561303370786517M,
        [|1.0; 3.0; 10.0; 0.68; 1.0; 0.0; -0.88; -0.84; -0.63; 0.0|]
        //(1.5295550561797753M,
        [|1.0; 3.0; 10.0; 0.81; 0.76; 0.0; -0.39; -0.6; -0.6; -0.01|]
        //(1.4852629213483146M,
        [|1.0; 3.0; 10.0; 1.0; 1.0; -0.3; -1.0; -0.86; -0.81; 0.0|]
        //(1.4852629213483146M,
        [|1.0; 3.0; 10.0; 0.97; 0.98; -0.29; -0.5; -0.63; -0.68; 0.0|]
        //(1.4350651685393259M,
        [|1.0; 3.0; 10.0; 1.0; 0.74; -0.12; 0.0; -0.71; -0.66; 0.0|]
        //(1.3937258426966292M,
        [|5.0; 6.0; 20.0; 0.64; 0.78; 0.0; 0.0; -0.7; -0.6; -0.02|]
    ]
    |> List.map (fun v -> 
        {TuneParms.Default with 
            Layers = int v.[0]
            Lookback = int v.[1]
            TrendWindowBars = int v.[2]
            GoodBuyInterReward = v.[3]
            GoodSellInterReward = v.[4]
            BadBuyInterPenalty = v.[5]
            BadSellInterPenalty = v.[6]
            ImpossibleBuyPenalty = v.[7]
            ImpossibleSellPenalty = v.[8]
            NonInvestmentPenalty = v.[9]    
        }
    )

let parmSpace = [0.001,tp]//; 0.001,8L; 0.001,10]///; 0.0001; 0.0002; 0.00001]
//let parms = parmSpace |> List.mapi (fun i ps -> parms1 i ps)
let parms = caSols |> List.mapi (fun i p -> parms1 i (0.001,p))



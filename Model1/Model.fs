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
    let slices,_ = Data.testTrain tp
    let episodes = Data.episodeLengthMarketSlices slices
    {Parms.Create createModel DQN lr episodes.Length id with 
        SyncEverySteps = 1000
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
        // 1.34159
        [|3.0; 33.33333333; 100.0; 0.006; 0.01; -0.003; 0.0; -0.01; -0.01; -0.001|]
        // 1.06953
        [|6.0; 133.3333333; 400.0; 0.0074; 0.0072; -0.0029; -0.01; -0.0094; -0.01;
        -0.001|]
        // 1.03842
        [|3.0; 66.66666667; 200.0; 0.0071; 0.0072; -0.0008; -0.006; -0.0072; -0.0085;
        -0.0004|]
        // 1.03073
        [|2.0; 33.33333333; 100.0; 0.0069; 0.0077; -0.0003; -0.0098; -0.0087; -0.0083;
        -0.0003|]
        // 1.01025
        [|2.0; 66.66666667; 200.0; 0.006; 0.0089; -0.0005; -0.01; -0.01; -0.0096; 0.0|]
        // 0.99745
        [|2.0; 33.33333333; 100.0; 0.01; 0.0071; -0.0019; -0.0039; -0.0068; -0.0088;
        -0.0009|]
        // 0.99554
        [|6.0; 66.66666667; 200.0; 0.0084; 0.0076; -0.0011; -0.01; -0.0089; -0.0074;
        -0.0008|]
        // 0.98040
        [|3.0; 66.66666667; 200.0; 0.0092; 0.0075; -0.003; -0.0062; -0.0067; -0.0078;
        0.0|]
        // 0.96839
        [|2.0; 33.33333333; 100.0; 0.006; 0.0076; 0.0; -0.009; -0.0099; -0.0082; -0.0002|]
        // 0.94143
        [|4.0; 33.33333333; 100.0; 0.0065; 0.0074; -0.0015; -0.0012; -0.0068; -0.0082;
        -0.0005|]
        // 1.34159
        [|3.0; 33.33333333; 100.0; 0.006; 0.01; -0.003; 0.0; -0.01; -0.01; -0.001|]
        // 1.06953
        [|6.0; 133.3333333; 400.0; 0.0074; 0.0072; -0.0029; -0.01; -0.0094; -0.01;
        -0.001|]
        // 1.03842
        [|3.0; 66.66666667; 200.0; 0.0071; 0.0072; -0.0008; -0.006; -0.0072; -0.0085;
        -0.0004|]
        // 1.03073
        [|2.0; 33.33333333; 100.0; 0.0069; 0.0077; -0.0003; -0.0098; -0.0087; -0.0083;
        -0.0003|]
        // 1.01025
        [|2.0; 66.66666667; 200.0; 0.006; 0.0089; -0.0005; -0.01; -0.01; -0.0096; 0.0|]
        // 0.99745
        [|2.0; 33.33333333; 100.0; 0.01; 0.0071; -0.0019; -0.0039; -0.0068; -0.0088;
        -0.0009|]
        // 0.99554
        [|6.0; 66.66666667; 200.0; 0.0084; 0.0076; -0.0011; -0.01; -0.0089; -0.0074;
        -0.0008|]
        // 0.98040
        [|3.0; 66.66666667; 200.0; 0.0092; 0.0075; -0.003; -0.0062; -0.0067; -0.0078;
        0.0|]
        // 0.96839
        [|2.0; 33.33333333; 100.0; 0.006; 0.0076; 0.0; -0.009; -0.0099; -0.0082; -0.0002|]
        // 0.94143
        [|4.0; 33.33333333; 100.0; 0.0065; 0.0074; -0.0015; -0.0012; -0.0068; -0.0082;
        -0.0005|]
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



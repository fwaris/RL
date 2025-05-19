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
        [|1.0; 3.0; 10.0; 0.7344444444; 0.9236111111; -0.1113888889; -0.5736111111;
        -0.7319444444; -0.8366666667; -0.04805555556|]
        [|1.0; 3.0; 10.0; 0.6822222222; 0.93; -0.2466666667; -0.2211111111;
        -0.7511111111; -0.7244444444; -0.03888888889|]
        [|1.0; 3.0; 10.0; 0.8108571429; 0.8388571429; -0.08685714286; -0.3897142857;
        -0.6402857143; -0.7351428571; -0.05628571429|]
        [|1.0; 3.0; 10.0; 0.6076923077; 0.7653846154; 0.0; -0.3661538462; -0.6;
        -0.9615384615; -0.03307692308|]
        [|1.0; 3.0; 10.0; 0.8841176471; 0.9952941176; -0.1117647059; 0.0; -0.8194117647;
        -0.6888235294; -0.08235294118|]
        [|2.208333333; 3.0; 10.0; 0.8322916667; 0.825; -0.07625; -0.3639583333;
        -0.6508333333; -0.7520833333; -0.048125|]
        [|1.0; 3.0; 10.0; 0.7717241379; 0.9044827586; -0.1562068966; -0.9548275862;
        -0.6448275862; -0.6917241379; -0.03551724138|]
        [|1.0; 3.0; 10.0; 0.8727272727; 0.8595454545; -0.07636363636; -0.03636363636;
        -0.6427272727; -0.9659090909; -0.06636363636|]
        [|1.0; 3.0; 10.0; 0.8878823529; 0.8477647059; -0.1252941176; -0.8732941176;
        -0.6236470588; -0.8385882353; -0.08|]
        [|2.0; 6.0; 20.0; 0.6766666667; 0.6566666667; -0.006666666667; -0.55; -0.76;
        -0.75; -0.04666666667|]
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



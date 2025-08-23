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
    let nheads = 4
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
                use dec = encB.[``:``,FIRST,``:``]    //keep first value as output to compare with target - B x emsize
                let pout = projOut.forward(dec) //B x ACTIONS
                //let act = activation.forward pout
                //let t_act = Tensor.getDataNested<float32> act
                pout
            )
        mdl
    let model = DQNModel.create createModel    
    let exp = {Decay=0.9995; Min=0.01; WarupSteps=WARMUP}
    let DQN = DQN.create model 0.99999f exp ACTIONS
    let slices = Data.getSlices tp
    let episodes = Data.numberOfOEpisodeLengthMarketSlices slices
    {Parms.Create createModel DQN lr episodes id with 
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

let solSet5 = [
    // 0.46770
    "t1",[|5.0; 1133.333333; 3400.0; 0.0; 0.0; -0.005; -0.0042; -0.01; -0.01; -0.0004|]
    // 0.46641
    "t2",[|2.0; 1266.666667; 3800.0; 0.0; 0.0064; -0.0099; -0.0076; -0.0068; -0.01;
      -0.0009|]
    // 0.44549
    "t3",[|2.0; 1933.333333; 5800.0; 0.0; 0.01; -0.005; -0.0003; -0.0072; -0.01; -0.001|]
    // 0.44215
    "t4",[|1.0; 833.3333333; 2500.0; 0.01; 0.0011; 0.0; -0.01; -0.006; -0.0096; -0.0004|]
    // 0.43807
    "t5",[|1.0; 1066.666667; 3200.0; 0.0; 0.0006; -0.0084; 0.0; -0.0092; -0.0072; 0.0|]
    // 0.43733
    "t6",[|1.0; 800.0; 2400.0; 0.01; 0.0024; -0.0043; -0.0015; -0.0098; -0.01; 0.0|]
    // 0.43554
    "t7",[|2.0; 1133.333333; 3400.0; 0.01; 0.0054; -0.01; -0.0043; -0.0081; -0.007; 0.0|]
    // 0.42928
    "t8",[|1.0; 800.0; 2400.0; 0.0033; 0.0098; -0.0046; -0.01; -0.01; -0.0072; 0.0|]
    // 0.42863
    "t9",[|1.0; 566.6666667; 1700.0; 0.0085; 0.0; -0.0039; -0.0095; -0.0068; -0.0026;
      -0.0003|]
    // 0.39806
    "t10",[|1.0; 966.6666667; 2900.0; 0.0; 0.0089; -0.0052; -0.0004; -0.01; -0.0069; 0.0|]  
]

let solSet4 = [
        "d1",[|3.866666794f; 1158.889038f; 3476.666748f; 0.006176668685f; 0.002513333224f;
            -0.005016666371f; -0.004763333593f; -0.008186667226f; -0.008683334105f;
            -0.0002000000386f|]
        "d2",[|3.0f; 448.4848328f; 1345.45459f; 0.00416363636f; 0.002381818369f;
          -0.00523636397f; -0.006399999373f; -0.007854545489f; -0.00848181732f;
          -0.0001818181772f|]
        "d3",[|1.75f; 108.3333282f; 325.0f; 0.002875000006f; 0.007649999578f; -0.00742499996f;
          -0.005349999759f; -0.008975000121f; -0.006650000345f; -7.499999629e-05f|]
        "d4",[|2.333333492f; 266.666687f; 800.0f; 0.004233333748f; 0.001899999916f;
          -0.007583332714f; -0.007483333349f; -0.007966667414f; -0.00800000038f;
          -0.0001000000047f|]
        "d5",[|2.409090996f; 798.4848633f; 2395.45459f; 0.005586363841f; 0.003095454536f;
          -0.006490909494f; -0.007059091236f; -0.008550000377f; -0.008672729135f;
          -0.0003636363545f|]
]

let solSet3 = [
        // 0.54443
        "a1",[|4.0; 1166.666667; 3500.0; 0.0095; 0.0023; -0.0045; 0.0; -0.0083; -0.0098; 0.0|]
        // 0.48983
        "a2",[|5.0; 766.6666667; 2300.0; 0.0045; 0.0037; -0.0086; -0.0096; -0.0083; -0.0087;
          -0.0006|]
        // 0.45128
        "a3",[|5.0; 1066.666667; 3200.0; 0.0021; 0.003; -0.0019; -0.007; -0.0067; -0.0098;
          -0.0002|]
        // 0.44847
        "a4",[|1.0; 933.3333333; 2800.0; 0.0093; 0.0029; -0.0075; -0.01; -0.0098; -0.01;
          -0.0006|]
        // 0.43643
        "a5",[|4.0; 633.3333333; 1900.0; 0.0031; 0.0038; -0.0065; -0.0096; -0.0084; -0.0067;
          -0.0004|]
        // 0.42880
        "a6",[|3.0; 1133.333333; 3400.0; 0.0076; 0.0009; -0.0046; -0.0006; -0.0081; -0.0083;
          0.0|]
        //****** best for now
        // 0.42197
        "a7",[|2.0; 300.0; 900.0; 0.0016; 0.0; -0.0095; -0.0087; -0.0089; -0.0088; 0.0|]
        // 0.38905
        "a8",[|3.0; 1133.333333; 3400.0; 0.0075; 0.0008; -0.0047; -0.0007; -0.0082; -0.0083;
          0.0|]
        // 0.38343
        "a9",[|4.0; 100.0; 300.0; 0.0027; 0.0075; -0.0099; -0.0077; -0.0089; -0.0053; -0.0001|]
        // 0.37138
        "a10",[|6.0; 1066.666667; 3200.0; 0.01; 0.0022; -0.0038; -0.0075; -0.0079; -0.0076;
          0.0|]    
]

let solSet2 = [
    // 0.54443
    "s1",[|4.0; 1166.666667; 3500.0; 0.0095; 0.0023; -0.0045; 0.0; -0.0083; -0.0098; 0.0|]
    // 0.48983
    "s2",[|5.0; 766.6666667; 2300.0; 0.0045; 0.0037; -0.0086; -0.0096; -0.0083; -0.0087;
      -0.0006|]
    // 0.45128
    "s3",[|5.0; 1066.666667; 3200.0; 0.0021; 0.003; -0.0019; -0.007; -0.0067; -0.0098;
      -0.0002|]
    // 0.44847
    "s4",[|1.0; 933.3333333; 2800.0; 0.0093; 0.0029; -0.0075; -0.01; -0.0098; -0.01;
      -0.0006|]
    // 0.43643
    "s5",[|4.0; 633.3333333; 1900.0; 0.0031; 0.0038; -0.0065; -0.0096; -0.0084; -0.0067;
      -0.0004|]
    // 0.42880
    "s6",[|3.0; 1133.333333; 3400.0; 0.0076; 0.0009; -0.0046; -0.0006; -0.0081; -0.0083;
      0.0|]
    // 0.42197
    "s7",[|2.0; 300.0; 900.0; 0.0016; 0.0; -0.0095; -0.0087; -0.0089; -0.0088; 0.0|]
    // 0.38905
    "s8",[|3.0; 1133.333333; 3400.0; 0.0075; 0.0008; -0.0047; -0.0007; -0.0082; -0.0083;
      0.0|]
    // 0.38343
    "s9",[|4.0; 100.0; 300.0; 0.0027; 0.0075; -0.0099; -0.0077; -0.0089; -0.0053; -0.0001|]
    // 0.37138
    "s10",[|6.0; 1066.666667; 3200.0; 0.01; 0.0022; -0.0038; -0.0075; -0.0079; -0.0076;
      0.0|]
]

let solSet1 = 
    [
        // 0.44847
        "a",[|1.0; 933.3333333; 2800.0; 0.0093; 0.0029; -0.0075; -0.01; -0.0098; -0.01;
            -0.0006|]
        // 0.43643
        "b",[|4.0; 633.3333333; 1900.0; 0.0031; 0.0038; -0.0065; -0.0096; -0.0084; -0.0067;
            -0.0004|]
        // 0.42880
        "c",[|3.0; 1133.333333; 3400.0; 0.0076; 0.0009; -0.0046; -0.0006; -0.0081; -0.0083;
            0.0|]
        //// 0.42197
        //[|2.0; 300.0; 900.0; 0.0016; 0.0; -0.0095; -0.0087; -0.0089; -0.0088; 0.0|]
        //// 0.38905
        //[|3.0; 1133.333333; 3400.0; 0.0075; 0.0008; -0.0047; -0.0007; -0.0082; -0.0083;
        //0.0|]
        //// 0.38343
        //[|4.0; 100.0; 300.0; 0.0027; 0.0075; -0.0099; -0.0077; -0.0089; -0.0053; -0.0001|]
        //// 0.37138
        //[|6.0; 1066.666667; 3200.0; 0.01; 0.0022; -0.0038; -0.0075; -0.0079; -0.0076;
        //0.0|]
        //// 0.37104
        //[|4.0; 566.6666667; 1700.0; 0.0041; 0.0028; -0.0064; -0.0012; -0.0088; -0.009;
        //-0.0002|]
    ]

let caSols =
    solSet5
    |> List.map (fun (id,v) -> 
        id,
        {TuneParms.Default with 
            Layers = int v.[0]
            Lookback = int v.[1]
            TrendWindowBars = int v.[2]
            GoodBuyInterReward = float v.[3]
            GoodSellInterReward = float v.[4]
            BadBuyInterPenalty = float v.[5]
            BadSellInterPenalty = float v.[6]
            ImpossibleBuyPenalty = float v.[7]
            ImpossibleSellPenalty = float v.[8]
            NonInvestmentPenalty = float v.[9]    
        }
    )

let parmSpace = [0.001,tp]//; 0.001,8L; 0.001,10]///; 0.0001; 0.0002; 0.00001]
//let parms = parmSpace |> List.mapi (fun i ps -> parms1 i ps)
let parms = lazy(caSols |> List.map (fun (id,p) -> parms1 id (0.001,p)))



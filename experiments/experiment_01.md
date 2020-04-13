**_Name_**: experiment_1
**_Location_**: `/data/Ponc/VOS-ForeGAN/experiment_1/`
**_Details_**: Train everything together at the same time, with 
    - lambda_rec  = 10
    - lambda_Db   = 10
    - lambda_Dbgp = 100

**_Results_**:
    :x: Gb does not learn how to impaint
    ![GitHub Logo](/experiments/imgs/experiment_01/bear_bg.gif)
    :wavy_dash: Gf learns slowly
    :heavy_check_mark: Gf learns something that resembles a real foreground [Fig. 1], the adversarial training is working on the foregrounds. Although this may seem a bad foreground, it models 'correctly' the distribution of black_bg+object.
    ![Github Logo](/experiments/imgs/experiment_01/bear_fg.png)
    :heavy_check_mark: Temporal skip connections work well! [Fig. 3]
    :x: Gf does not know how to capture the movement of the object, if we look at the back-left leg of the bear, is not modelled well.
    :x: L1 loss reaches a plateau

**_Conclusions_**:
    - Increase the number of patches to make Gb inpaint the image [Experiment 2]
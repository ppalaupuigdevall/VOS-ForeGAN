### **_Name_** 
experiment_2

### **_Location_** 
`/data/Ponc/VOS-ForeGAN/experiment_2/`

### **_Details_**
* We try to increase the number of patches/image in Db with N = 40 patches with size (60,112) and strides (10,12). We expect that the background distribution is better modeled than experiment_01. 

- lambda_rec  = 1
- lambda_Db   = 10
- lambda_Dbgp = 10

### **_Results_**

:x: Gf did not generate a real foreground (black + obj)

![GitHub Logo](/experiments/imgs/experiment_02/blackswan_fg.png)

:x: The mask took a lot of the foreground

![GitHub Logo](/experiments/imgs/experiment_02/mask_bear.png)

:x: Gb did not generate a background that fill all the image
![GitHub Logo](/experiments/imgs/experiment_02/fg_breakdance.png)

### **_Conclusions_**
- As a consequence of having increased the number of patches and lambda_Db, lambda_Dbgp, Gf did not generate real fgs, this fact misled Gb. In _experiment 3_ we will lower lambda_Db, lambda_Dbgp and set higher lambda_rec, lambda_Df, lambda_Dfgp. 
- Gf generates very strange things, this is probably a problem  of how C and A are combined, in this experiment:

A * (Iw1 + C) + [ (1 - A)*(Iw1 + C) - 1 ]

In this expression, we are expecting (Iw1 + C) to be the modified image and A to be directly the foreground mask. This is too much to ask from the model Gf. We'll modify this combination of C and A in experiment 3. 

- We'll keep with a lot of patches per image in the background.

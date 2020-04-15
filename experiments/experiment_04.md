### **_Name_** 
experiment_4

### **_Location_** 
`/data/Ponc/VOS-ForeGAN/experiment_4/`

### **_Details_**
- Until this experiment, we had been generating the fgmasks with the A mask. A conslusion of experiment 3 stated that this is not the correct way of generating a foreground mask since the attention mask looks where the warped image should be modified. In a diagram, the way of generating these foreground masks is depicted here:
![GitHub Logo](/experiments/imgs/experiment_04/experiment_1_2_3_4_mask.png)

- We keep the way of combining the A and C in Gf, but now we try to generate a mask out of the Ifg1 generated by Gf with a very simple conv layer (3..1) shared across timesteps.
![GitHub Logo](/experiments/imgs/experiment_04/experiment_5_fgmask.png)

- We increase T = 6, reduce N = 30
- We reduce a lot the size of the last feature map

### **_Results_**

:x: The quality of the generated images is worse. 

:heacy_check_mark: I discovered why part of the foreground is in the background after several training epochs. If we look at some generated foregreounds and backgrounds, we can see that the foreground is a little bit brighter than the background. This is because the background starts with a scene and a black region. Because of this black region, the network finds it easier to copy the dark values from the background and make a combination of the fg and bg to generate the foreground. In the next experiments, we will replace the black for gaussian noise in the black region of the background to help Gf generating the whole foreground.
<img align="left" width="500" height="272" src="https://github.com/ppalaupuigdevall/VOS-ForeGAN/tree/master/experiments/imgs/experiment_04/blackswan_bg_gif.gif">
<img align="right" width="500" height="271" src="https://github.com/ppalaupuigdevall/VOS-ForeGAN/tree/master/experiments/imgs/experiment_04/blackswan_fg.gif">

:x: The generated masks, as a consequence of the black background, do not mask all the foreground, just the bright parts. To solve this, we'll use a saturated sigmoid over the generated mask.
![GitHub Logo](/experiments/imgs/experiment_04/cow_mask.png)

### **_Conclusions_**
- The mask that is used to mask the background to generate should be created from Ifg1 mask it with a sigmoid.
- We have to replace tha black region of the background for noise.
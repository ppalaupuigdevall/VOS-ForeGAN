### **_Name_** 
experiment_6

### **_Details_**
The main scheme of how the system works is the following:

![GitHub Logo](/experiments/imgs/experiment_06/main_scheme_full.png)

Problems of this approach:
 - Gb is not receiving any temporal information so is only learning motion from MSE. This won't work at test time, we'll include OF as input to Gb in the next experiment or even a warped background of image 1.  
 - The background discriminator only analyzes Nr = 10, Nf = 10 patches of the background for each video at every epoch, that is the reason why it learns so slowly. We have a trade off between increasing T, Nr, Nf. If we increase T, we run out of memory in the GPU so we have to keep Nr, Nf low.
 - We commented with Octavia that we may include It+1 to the input of Gf because asking for a generated foreground of t+1 given the previous foreground and OF is too much.

 ### **_Results_**

 
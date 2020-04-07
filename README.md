# GANs

### TODO

 - [x] (Easy) Write architecture patchgan (1h 15m)
 - [x] (Normal) Write function that returns patches
 - [x] (Difficult) Make G recurrent
 - [x] (Easy) Modify Generator's forward method to concat warps, ofs, images. Header: forward(self, img0m, of, img1w)
 - [x] (Easy) Set inputs method in ForestGAN
 - [x] (Easy) Mask warped images
 - [x] (Normal) Implement forward of the whole GAN
 - [x] (Difficult) Implement forward D
 - [x] (Normal & urgent) Write extract_real_patches so it gives REAL patches where there's no foreground
 - [x] (Ultra difficult) Implement gradient penalty and optimize parameters

### Setup

```
source ../venmom/bin/activate
export PYTHONPATH=$PWD:/home/ppalau/VOS-ForeGAN/VFG/
```

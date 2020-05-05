# GANs

Results in experiments folder

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
 - [x] (Difficult) Why part of the generator is not in the GPU?
 - [x] Create mask from sigmoid saturated sigmoid( alpha * (x  + (1-eps) ) )
 - [x] Remove OF from input
 - [x] Create visualizations
 - [x] Clean generate_random_samples
 - [ ] Share weights between masks and fgs
### Setup

```
source ../venmom/bin/activate
export PYTHONPATH=$PWD:/home/ppalau/VOS-ForeGAN/VFG/
```
```
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
    
    
        
        
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "cwd": "/home/ppalau/TimeCycle-Dynamic-Tracking/SiamMask/tools/",
            "program": "${file}",
            "console": "integratedTerminal",
            "env": {"PYTHONPATH":"/home/ppalau/TimeCycle-Dynamic-Tracking/SiamMask/experiments/siammask_sharp:/home/ppalau/TimeCycle-Dynamic-Tracking/SiamMask/experiments/siammask_sharp:/home/ppalau/TimeCycle-Dynamic-Tracking/SiamMask"},
            "args":["--resume","../experiments/siammask_sharp/SiamMask_DAVIS.pth","--config","../experiments/siammask_sharp/config_davis.json","--base_path","/data/Ponc/tracking/JPEGImages/480p/nhl/"],
            
        },
        {
            "name": "Python: Prova file",
            "type": "python",
            "request": "launch",
            "cwd": "/home/ppalau/TimeCycle-Dynamic-Tracking/SiamMask/tools/",
            "program": "${file}",
            "console": "integratedTerminal",
            "env": {"PYTHONPATH":"/home/ppalau/TimeCycle-Dynamic-Tracking/SiamMask/experiments/siammask_sharp:/home/ppalau/TimeCycle-Dynamic-Tracking/SiamMask/experiments/siammask_sharp:/home/ppalau/TimeCycle-Dynamic-Tracking/SiamMask"},
            // "args":["--resume","../experiments/siammask_sharp/SiamMask_DAVIS.pth","--config","../experiments/siammask_sharp/config_davis.json","--base_path","/data/Ponc/tracking/JPEGImages/480p/nhl/"],
            
        }
    ]
}
```

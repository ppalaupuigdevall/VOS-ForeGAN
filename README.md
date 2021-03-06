# Self-Supervised Video Object Segmentation using Generative Adversarial Networks

This repository holds my M. Sc. Thesis done in <a href="http://robustsystems.coe.neu.edu/">Robust Systems Lab</a> at <b>Northeastern University </b>, Boston MA.

## Abstract

Video Object Segmentation is arguably one of the most challenging tasks of computer vision. Training a model in a supervised manner in this task requires a high number of manually labelled data, which is extremely time consuming and expensive to generate. In this thesis, we propose a self-supervised method that leverages the spatiotemporal nature of video to perform Video Object Segmentation using Generative Adversarial Networks. In this context, we design a novel framework composed of two generators and two discriminators that try to reach an equilibrium to fulfill the task. Both at training and testing time, the model needs only the first mask of the video to perform the task, which is possible because the model exploits the temporal consistency of videos to self-supervise its training. In addition, we refine the masks predicted by the model with the Sum of Squares polynomial, a tool adopted from convex optimization community. Although our approach is considerably ambitious, our model achieves promising results on DAVIS2016 dataset, which are reported in a qualitative and quantitative manner.

## Results
Main results at time horizon 3T of our models: _baseline_, _Dmask_ and _Dmask-prop_.


                Baseline                            Dmask                           Dmask-prop
<img src="/results/imgs/baseline/training/baseline_bmxbumps.gif" width="260" height="140"/> <img src="/results/imgs/dmask/training/dmask_bmxbumps.gif" width="260" height="140"/> <img src="/results/imgs/dmask-prop/training/dmaskprop_bmxbumps.gif" width="260" height="140"/>
<img src="/results/imgs/baseline/training/baseline_motocrossbumps.gif" width="260" height="140"/> <img src="/results/imgs/dmask/training/dmask_motocrossbumps.gif" width="260" height="140"/> <img src="/results/imgs/dmask-prop/training/dmaskprop_motocrossbumps.gif" width="260" height="140"/>
<img src="/results/imgs/baseline/training/baseline_tennis.gif" width="260" height="140"/> <img src="/results/imgs/dmask/training/dmask_tennis.gif" width="260" height="140"/> <img src="/results/imgs/dmask-prop/training/dmaskprop_tennis.gif" width="260" height="140"/>

### Background and Foreground cooperation
<img src="/results/imgs/dmask-prop/training/bg_rollerblade.gif" width="260" height="140"/> <img src="/results/imgs/dmask-prop/training/dmaskprop_rollerblade.gif" width="260" height="140"/>

### Validation results


                Baseline                            Dmask                           Dmask-prop
<img src="/results/imgs/baseline/validation/baseline_horsejump.gif" width="260" height="140"/> <img src="/results/imgs/dmask/validation/dmask_horsejump.gif" width="260" height="140"/> <img src="/results/imgs/dmask-prop/validation/dmaskprop_horsejump.gif" width="260" height="140"/>


## Setup

```
source ../venmom/bin/activate
export PYTHONPATH=$PWD:/home/ppalau/VOS-ForeGAN/VFG/
```

```
python VFG/generate_random_samples.py --model forestgan_rnn_v10 --name forestgan_rnn_v10_BF_sastig_20_baseline  --load_epoch 2000 --gpu_ids 0 --T 26

python VFG/generate_random_samples.py --model forestgan_rnn_v10 --name forestgan_rnn_v10_BF_sastig_20_baseline  --load_epoch 2000 --gpu_ids 0 --T 26

python VFG/generate_random_samples.py --model forestgan_rnn_v10 --name forestgan_rnn_v10_BF_sastig_20_baseline  --load_epoch 2000 --gpu_ids 0 --T 26
```
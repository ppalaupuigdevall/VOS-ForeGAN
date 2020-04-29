import os 
import numpy as np
import pyflow
import pandas as pd
from PIL import Image



alpha = 0.012
ratio = 0.75
minWidth = 20
nOuterFPIterations = 7
nInnerFPIterations = 1
nSORIterations = 30
colType = 0

saveDir = '/data/Ponc/DAVIS/newFlows/'
rootDir = '/data/Ponc/DAVIS/JPEGImages/480p/'
folder = 'rollerblade'

frames = [each for each in os.listdir(os.path.join(rootDir,folder)) if each.endswith(('.jpg','.jpeg'))]
nFrames = len(frames)
frames.sort()
save_flows = None
for framenum in range(0,nFrames-1):
    imgname = os.path.join(rootDir,folder,frames[framenum])
    img1 = np.array(Image.open(imgname))/255.
    
    imgname = os.path.join(rootDir,folder,frames[framenum+1])
    img2 = np.array(Image.open(imgname))/255.
    
    u, v, img2W = pyflow.coarse2fine_flow( img2, img1, alpha, ratio, minWidth, 
                        nOuterFPIterations, nInnerFPIterations, nSORIterations, colType)
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    if not os.path.exists(os.path.join(saveDir, folder)):
        os.makedirs(os.path.join(saveDir, folder))
    np.save(os.path.join(saveDir, folder,"{:05d}".format(framenum)+'.npy'), flow)
  

    if(framenum==0):
        save_flows = True
    if(save_flows):
        import cv2
        hsv = np.zeros(img1.shape, dtype=np.uint8)
        hsv[:, :, 0] = 255
        hsv[:, :, 1] = 255
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imwrite('/home/ppalau/VOS-ForeGAN/imgs/outFlow_new.png', rgb)
        cv2.imwrite('/home/ppalau/VOS-ForeGAN/imgs/car2Warped_new.jpg', img2W[:, :, ::-1] * 255)
        save_flows = False
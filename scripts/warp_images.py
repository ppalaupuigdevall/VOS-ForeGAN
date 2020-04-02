import cv2
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import argparse
import os
import matplotlib.pyplot as plt
from utils_f.utils_flow import warp_flow
from utils_f.utils_flow import draw_flow
from utils_f.utils_flow import readFlow

def resize_img(img, size):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    resi = transforms.Compose([transforms.Resize(size)])
    img_pil_resized = resi(img_pil)
    img_cv2_resized = cv2.cvtColor(np.array(img_pil_resized), cv2.COLOR_RGB2BGR)
    return img_cv2_resized

def resize_gray_img(img, size):
    img_pil = Image.fromarray(img)
    resi = transforms.Compose([transforms.Resize(size)])
    img_pil_resized = resi(img_pil)
    img_cv2_resized = np.array(img_pil_resized)
    return img_cv2_resized

def remap_values(values, xmin, xmax, ymin, ymax):
    """
    Remaps values with a positive straight line (es podria posar com a parametre si fos ppendent >0 o <0)
    """
    values = np.clip(values, xmin, xmax)
    m = (ymax - ymin)/(xmax - xmin)
    n = m * xmin
    values = np.uint8(m * values + n)
    return values

#https:///GANimation/issues/22 (utilitzen 128 x 128)

resolutions = {0:(140,260), 1:(175,325), 2:(224,416), 3:(280, 520)}
desired_shape = resolutions[2]

img_dir = '/data/Ponc/DAVIS/JPEGImages/480p/bus/'
flo_dir = '/data/Ponc/DAVIS/OpticalFlows/bus/'
mask_dir = '/data/Ponc/DAVIS/Annotations/480p/bus/'

img_name = '00000.jpg'
flo_name = '000000.flo'
mask_name = '00000.png'



flow = readFlow(os.path.join(flo_dir, flo_name))
img = cv2.imread(os.path.join(img_dir, img_name))
img_warped = warp_flow(img, flow)
mask = cv2.imread(os.path.join(mask_dir, mask_name))
mask_bg = cv2.bitwise_not(mask)
print(mask.shape)
masked_img = cv2.bitwise_and(img, mask)

I_1_warped = warp_flow(masked_img, flow)
flo_name = '000001.flo'
flow = readFlow(os.path.join(flo_dir, flo_name))
I_2_warped = warp_flow(I_1_warped, flow)

flo_name = '000002.flo'
flow = readFlow(os.path.join(flo_dir, flo_name))
I_3_warped = warp_flow(I_2_warped, flow)

I_1_warped_rsz = resize_img(I_1_warped, desired_shape)
I_2_warped_rsz = resize_img(I_2_warped, desired_shape)
I_3_warped_rsz = resize_img(I_3_warped, desired_shape)

masked_img_bg = cv2.bitwise_and(img, mask_bg)
u = flow[:,:,0]
v = flow[:,:,1]
flow_u_remaped = remap_values(u, -20, 20, 0, 255)
flow_v_remaped = remap_values(v, -20, 20, 0, 255)

# Resize ori to warped's shape
desired_shape = img_warped.shape[:2]
img_ori_resized = resize_img(img, desired_shape)
masked_img_resized = resize_img(masked_img, desired_shape)
masked_bg_resized = resize_img(masked_img_bg, desired_shape)
img_warped_resized = resize_img(img_warped,desired_shape)
flow_u_remaped_resized = resize_gray_img(flow_u_remaped, desired_shape)

display, save = False, True

if(display):
    cv2.imshow('v', flow_v_remaped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if(save):
    cv2.imwrite('./imgs/warp1.jpg', img_warped)
    cv2.imwrite('./imgs/img_warped_resized.jpg', img_warped_resized)
    cv2.imwrite('./imgs/img_ori_resized.jpg',img_ori_resized)
    cv2.imwrite('./imgs/flow_u_resized.jpg', flow_u_remaped_resized)
    cv2.imwrite('./imgs/masked_img.jpg', masked_img_resized)
    cv2.imwrite('./imgs/masked_img_bg.jpg', masked_bg_resized)
    cv2.imwrite('./imgs/I1_warp.jpg', I_1_warped_rsz)
    cv2.imwrite('./imgs/I2_warp.jpg', I_2_warped_rsz)
    cv2.imwrite('./imgs/I3_warp.jpg', I_3_warped_rsz)
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

img_dir = '/data/Ponc/DAVIS/JPEGImages/480p/camel/'
flo_dir = '/data/Ponc/DAVIS/newFlows/camel/'
mask_dir = '/data/Ponc/DAVIS/Annotations/480p/camel/'

img_list = sorted(os.listdir(img_dir))

img_name = '00000.jpg'
flo_name = '00000.npy'
mask_name = '00000.png'


# flow = np.load(os.path.join(flo_dir, flo_name))

img = cv2.imread(os.path.join(img_dir, img_name))
mask = cv2.imread(os.path.join(mask_dir, mask_name))
mask_bg = cv2.bitwise_not(mask)

mask_uni = mask[:,:,0]
new_img = img.copy()
for i in range(img.shape[0]):
    idxs = np.where(mask_uni[i,:]==255)[0]
    if(len(idxs)>0):
        print(idxs)
        num_idxs = len(idxs)
        new_img[i,idxs[0:num_idxs//2],:] = img[i,idxs[0]-3,:]
        new_img[i, idxs[num_idxs//2 +1 :], :] = img[i, idxs[-1]+3,:]

cv2.imwrite('./imgs/new/new_img.jpg', new_img) 
img_warped = warp_flow(img, flow)
masked_fg = cv2.bitwise_and(img, mask)
noise = np.random.normal(0,1,masked_fg.shape) *255 - 127
noise = np.uint8(noise)
masked_noise = cv2.bitwise_and(mask, noise)

masked_bg = cv2.bitwise_and(img, mask_bg)
masked_bg_noise = resize_img(masked_bg +  masked_noise, desired_shape)
bg_warped = warp_flow(masked_bg, flow)

I_1_warped = warp_flow(masked_fg, flow)
I_bg_1_warp = warp_flow(masked_bg, flow)
flo_name = '00001.npy'
flow = np.load(os.path.join(flo_dir, flo_name))
I_2_warped = warp_flow(I_1_warped, flow)
I_bg_2_warp = warp_flow(I_bg_1_warp, flow)
flo_name = '00002.npy'
flow = np.load(os.path.join(flo_dir, flo_name))
I_3_warped = warp_flow(I_2_warped, flow)
I_bg_3_warp = warp_flow(I_bg_2_warp, flow)
flo_name = '00003.npy'
flow = np.load(os.path.join(flo_dir, flo_name))
I_4_warped = warp_flow(I_3_warped, flow)
I_bg_4_warp = warp_flow(I_bg_3_warp, flow)

I_1_warped_rsz = resize_img(I_1_warped, desired_shape)
I_2_warped_rsz = resize_img(I_2_warped, desired_shape)
I_3_warped_rsz = resize_img(I_3_warped, desired_shape)
I_4_warped_rsz = resize_img(I_4_warped, desired_shape)

flo_name = '00000.npy'
flow = np.load(os.path.join(flo_dir, flo_name))
u = flow[:,:,0]
v = flow[:,:,1]
flow_u_remaped = remap_values(u, -35, 35, 0, 255)
flow_v_remaped = remap_values(v, -35, 35, 0, 255)


# Resize ori to warped's shape
# desired_shape = img_warped.shape[:2]
img_ori_resized = resize_img(img, desired_shape)
masked_fg_resized = resize_img(masked_fg, desired_shape)
masked_bg_resized = resize_img(masked_bg, desired_shape)
img_warped_resized = resize_img(img_warped,desired_shape)
flow_u_remaped_resized = resize_gray_img(flow_u_remaped, desired_shape)
flow_v_remaped_resized = resize_gray_img(flow_v_remaped, desired_shape)
masked_bg_noise_resized = resize_img(masked_bg_noise,desired_shape)
display, save = False, False

list_of_resized_ims = []
for i in range(7):
    list_of_resized_ims.append(resize_img(cv2.imread(os.path.join(img_dir, img_list[i])), desired_shape))

if(display):
    cv2.imshow('v', flow_v_remaped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if(save):
    cv2.imwrite('./imgs/new/warp1.jpg', img_warped)
    cv2.imwrite('./imgs/new/img_warped_resized.jpg', img_warped_resized)
    cv2.imwrite('./imgs/new/img_ori_resized.jpg',img_ori_resized)
    cv2.imwrite('./imgs/new/flow_u_resized.jpg', flow_u_remaped_resized)
    cv2.imwrite('./imgs/new/flow_v_resized.jpg', flow_v_remaped_resized)
    cv2.imwrite('./imgs/new/masked.jpg', masked_fg_resized)
    cv2.imwrite('./imgs/new/masked_bg.jpg', masked_bg_resized)
    cv2.imwrite('./imgs/new/I1_warp.jpg', I_1_warped_rsz)
    cv2.imwrite('./imgs/new/I2_warp.jpg', I_2_warped_rsz)
    cv2.imwrite('./imgs/new/I3_warp.jpg', I_3_warped_rsz)
    cv2.imwrite('./imgs/new/I4_warp.jpg', I_4_warped_rsz)
    cv2.imwrite('./imgs/new/I_bg_1_warp.jpg', I_bg_1_warp)
    cv2.imwrite('./imgs/new/I_bg_2_warp.jpg', I_bg_2_warp)
    cv2.imwrite('./imgs/new/I_bg_3_warp.jpg', I_bg_3_warp)
    cv2.imwrite('./imgs/new/I_bg_4_warp.jpg', I_bg_4_warp)
    cv2.imwrite('./imgs/new/masked_noise.jpg', masked_noise)
    cv2.imwrite('./imgs/new/masked_bg_noise.jpg', masked_bg_noise_resized)
    cv2.imwrite('./imgs/new/bg_warped.jpg', bg_warped)
    for i in range(7):
        cv2.imwrite('./imgs/new/'+str(i)+'.jpg', list_of_resized_ims[i])
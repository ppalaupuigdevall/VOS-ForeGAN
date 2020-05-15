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
import torch

def tensor2im(img, imtype=np.uint8, unnormalize=True, idx=0, nrows=None):
    # select a sample or create grid if img is a batch
    if len(img.shape) == 4:
        nrows = nrows if nrows is not None else int(math.sqrt(img.size(0)))
        img = img[idx] if idx >= 0 else torchvision.utils.make_grid(img, nrows)

    img = img.cpu().float()
    if unnormalize:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

        for i, m, s in zip(img, mean, std):
            i.mul_(s).add_(m)

    image_numpy = img.numpy()
    image_numpy_t = np.transpose(image_numpy, (1, 2, 0))
    image_numpy_t = image_numpy_t*254.0

    return image_numpy_t.astype(imtype)



def resize_img(img, size):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    resi = transforms.Compose([transforms.Resize(size)])
    img_pil_resized = resi(img_pil)
    # img_cv2_resized = cv2.cvtColor(np.array(img_pil_resized), cv2.COLOR_RGB2BGR)
    # return img_cv2_resized
    return img_pil_resized

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
categ = 'mallard-fly'
img_dir = '/data/Ponc/DAVIS/JPEGImages/480p/training/' + categ +'/'
flo_dir = '/data/Ponc/DAVIS/OpticalFlows/training/'+categ+'/'
mask_dir = '/data/Ponc/DAVIS/Annotations/480p/'+categ+'/'

img_list = sorted(os.listdir(img_dir))

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

geom_transforms = [
    transforms.RandomAffine(5,(0,0.02),(0.75,1.25)),transforms.RandomHorizontalFlip(0.5) 
]
to_grayscale = transforms.Grayscale()
to_tensor = transforms.ToTensor()
norma=transforms.Normalize(mean=0.5,\
                                 std=0.5)
transform_img = transforms.RandomApply(geom_transforms, p=0.45)
gray=to_grayscale(resize_img(img, (224,416)))

gray_tensor = norma(to_tensor(gray))

print(tensor_mask.size()) # [3,224,416]
tensor_mask = norma(to_tensor(transform_img(resize_img(mask,(224,416)))))
tensor_mask = tensor_mask + (1-tensor_mask)*-1

mask_back = tensor2im(tensor_mask)
cv2.imwrite('./mask_ori.png', mask)
cv2.imwrite('./transformed_mask.png', mask_back)
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

def resize_flow(flow, size):
    
    return ''


#https:///GANimation/issues/22

resolutions = {0:(140,260), 1:(175,325), 2:(224,416), 3:(280, 520)}

img_dir = '/data/Ponc/DAVIS/JPEGImages/480p/bear/'
flo_dir = '/data/Ponc/DAVIS/OpticalFlows/bear/'

img_name = '00000.jpg'
flo_name = '000000.flo'

flow = readFlow(os.path.join(flo_dir, flo_name))
# Jo crec que el millor que es pot fer es en comptes de normalitzar el flow entre -1 i 1 (dividint entre 448,832), 
# dividir entre 255 per deixar-ho mes o menys al nivel de la imatge (com fan al lucid)
u = flow[:,:,0]/255.0
v = flow[:,:,1]/255.0
u_max = u/np.max(np.abs(u)) 
v_max = v/np.max(np.abs(v))



print(np.mean(u/448))
print(np.mean(v/832))
# u_128 = np.uint8((u + 1.0)*255)
# cv2.imshow('u',u_128)
# cv2.waitKey(0)
fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].imshow(u_max)
ax[1].imshow(v_max)
plt.show()

img = cv2.imread(os.path.join(img_dir, img_name))
img_warped = warp_flow(img, flow)

print(img.shape)
print(img_warped.shape)
print(flow.shape)

# Resize ori to warped's shape
desired_shape = img_warped.shape[:2]
desired_shape = resolutions[2]

img_ori_resized = resize_img(img, desired_shape)
img_warped_resized = resize_img(img_warped,desired_shape)

cv2.imwrite('./warp1.jpg', img_warped)
cv2.imwrite('./img_warped_resized.jpg', img_warped_resized)
cv2.imwrite('./img_ori_resized.jpg',img_ori_resized)
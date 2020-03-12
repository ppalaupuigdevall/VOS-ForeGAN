import cv2
import numpy as np
import argparse
import os
from utils_f.utils_flow import warp_flow
from utils_f.utils_flow import draw_flow
from utils_f.utils_flow import readFlow

img_dir = '/data/Ponc/DAVIS/JPEGImages/480p/bear/'
flo_dir = '/data/Ponc/DAVIS/OpticalFlows/bear/'

img_name = '00000.jpg'
flo_name = '000050.flo'

flow = readFlow(os.path.join(flo_dir, flo_name))

img = cv2.imread(os.path.join(img_dir, img_name))
img_warped = warp_flow(img, flow)

cv2.imwrite('./test.jpg', img_warped)
cv2.imwrite('./test_2.jpg', res_img)

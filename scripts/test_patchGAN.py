import cv2
import numpy as np


img = cv2.imread('./masked_img.jpg')
img = img[:,:,:]
sx, sy = 64, 64
kernel = np.ones((sx,sy),np.float32)
dst = cv2.filter2D(img,-1,kernel)
dst = dst[sx//2:-sx//2, sy//2:-sy//2]
rows, cols,c = dst.shape
print(rows)
print(cols)

for i in range(rows):
    
    for j in range(cols):
        if(dst[i,j,0]==0.0):
            cv2.rectangle(img,(j,i), (j+sx,i+sy), (255,0,0))
            cv2.imshow('a', img)
            cv2.waitKey(1)
            cv2.rectangle(img,(j,i), (j+sx,i+sy), (0,0,0))
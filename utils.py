import torch
from torchvision import transforms, datasets, utils
from PIL import Image
import numpy as np
import math


def norm_noise(size):
    # Create (size)x(100) matrix containing all the z random noise vectors of all the batch (size = batchsize)
    return torch.cuda.FloatTensor(size, 100).normal_()


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('BatchNorm') != -1:
        m.weight.data.normal_(0.00, 0.02)


def display_batch_images(img, imtype=np.uint8, unnormalize=True, nrows=None, mean=0.5, std=0.5):
    # Create grid if img is a batch of images (len = 4)
    if len(img.shape) == 4:
        nrows = nrows if nrows is not None else int(math.sqrt(img.size(0)))
        img = utils.make_grid(img, nrow=nrows)

    # Unnormalize Image
    img = img.cpu().float()
    img = (img*std+mean)*255

    return img

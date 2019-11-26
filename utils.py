import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets, utils
from PIL import Image
import numpy as np
import math


def norm_noise(size):
    # Create (size)x(100) matrix containing all the z random noise vectors of all the batch (size = batchsize)
    return torch.cuda.FloatTensor(size, 100).normal()


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('BatchNorm') != -1:
        m.weight.data.normal_(0.00, 0.02)


def display_batch_images(img, imtype=np.unit8, unnormalize=True, nrows=None, mean=0.5, std=0.5):
    # Select a sample or create grid if img is a batch
    if len(img.shape) == 4:
        nrows = nrows if nrows is not None else int(math.sqrt(img.size(0)))
        img = utils.make_grid(img, nrow=nrows)

    # Unnormalize
    img = img.cpu().float()
    img = (img*std+mean)*255

    # Pass to NumPy
    image_numpy = img.numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    # display(Image.fromarray(image_numpy.astype(imtype)))
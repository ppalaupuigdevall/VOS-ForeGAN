import torch
from torchvision import models
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import requests
from io import BytesIO

net = torch.hub.load('pytorch/vision:v0.5.0', 'deeplabv3_resnet101', pretrained=True)
# model2 = models.segmentation.deeplabv3_resnet101(pretrained=True)
# model1.eval()

def load_img_from_url(url):
  response = requests.get(url)
  img = Image.open(BytesIO(response.content))
  return img


def preprocess_img(img, device):
  # img H x W x 3
  transform = transforms.Compose([
    transforms.Resize(256),  #  H' x W' x 3
    transforms.ToTensor(),   #  3 x H' x W'
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])

  input_batch = transform(img)  # 1 x 3 x H' x W'
  input_batch = input_batch.unsqueeze(0)
  input_batch.to(device)
  return input_batch


def create_labels_palette(num_classes=21):
  r = np.linspace(0, num_classes-1, num_classes)
  g = np.linspace(0, 255, num_classes)
  b = np.linspace(0, 255, num_classes//3).repeat(3)[:num_classes]
  colors = np.stack([r,g,b],1).astype("uint8")
  return colors


def plot_labels(img_labels, palette):
  img = Image.fromarray(img_labels.byte().cpu().numpy()).resize(input_image.size)
  img.putpalette(palette)
  plt.imshow(img)


input_image = load_img_from_url('https://raw.githubusercontent.com/pytorch/hub/master/dog.jpg')
filename = 'dog.jpg'
filename = Image.open(filename)

# preprocess img
device = "cuda" if torch.cuda.is_available() else "cpu"
input_batch = preprocess_img(filename, device)

# estimate
net.eval()
estim = net(input_batch)['out'][0]  # num_classes x H x W
estim_class = estim.argmax(0)       # H x W

palette = create_labels_palette()
img = Image.fromarray(estim_class.byte().cpu().numpy()).resize(input_image.size)
img.putpalette(palette)

img.save('ima5.png')

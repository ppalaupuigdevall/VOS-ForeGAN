import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

# imports pel dataset
import os
from PIL import Image
class FirstMaskDavis(torch.utils.data.Dataset):
    """
    NOTE: Aixo hauria d'anar a un altre file pero ja ho farem
    """
    def __init__(self, root_folder = '/data/DAVIS/', transfor = None):
        super(FirstMaskDavis, self).__init__()

        images_folder = os.path.join(root_folder, 'JPEGImages/480p')
        annots_folder = os.path.join(root_folder, 'Annotations/480p')
        categories = os.listdir(images_folder)
        self.num_videos = len(categories)

        self.images = []
        self.annots = []
        img = "{:05d}.jpg".format(0)
        ann = "{:05d}.png".format(0)
        for category in categories:
            self.images.append(os.path.join(images_folder, category, img))
            self.annots.append(os.path.join(annots_folder, category, ann))    

        self.transforms_images = transfor
        self.transforms_annots = torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)), transforms.ToTensor()])
    
    def __getitem__(self, idx):
        img = self.transforms_images(Image.open(self.images[idx]))
        # NOTE MOLT IMPORTANT: Les labels haurien de tindre la forma de H,W (on C=2 per FG/BG) (Ja ho son es nomes recordatori)
        ann = self.transforms_annots(Image.open(self.annots[idx]))   
        return (img, ann)

    def __len__(self):
        # There is only one mask for video so the length is the num of videos
        return self.num_videos


# Parameters
num_class = 2  # FG vs BG
batch_size = 8
num_epochs = 2
feature_extract = True  # True: only update reshaped layer params, False: finetune whole model
input_size = 224

preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

erdatase = FirstMaskDavis(root_folder='/data/DAVIS/', transfor = preprocess)
erdatalo = torch.utils.data.DataLoader(erdatase, batch_size=4, shuffle=False, num_workers=4)
guacamole = next(iter(erdatalo))
print(guacamole[0].size())

# NOTE NOTE NOTE !!! 
"""
Quan es fa segmentation, l'arquitectura treura un output de (BatchSize, num_classes, H, W)
De totes maneres, la label es una imatge de (H,W) on cada pixel te L'INDEX de la classe a la que pertany. 
En training, farem servir la nn.CrossEntropyLoss per compararlos. 
"""

# Load deep lab
model = torch.hub.load('pytorch/vision:v0.5.0', 'deeplabv3_resnet101', pretrained=True)
print(model.aux_classifier[4])
# Change dim of last layer to be num_classes
model.aux_classifier[4] = nn.Conv2d(256, 2, kernel_size=(1,1), stride=(1,1))
#Now copy parameters of all layers except last one
print(model.aux_classifier[4])




# # DataLoader
# data_dir = '/data/DAVIS/JPEGImages/480p/'
# # Data augmentation and normalization for training
# data_transforms = {
#     'train': transforms.Compose([
#         # transforms.RandomResizedCrop(input_size),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
#     'val': transforms.Compose([
#         # transforms.Resize(input_size),
#         transforms.CenterCrop(input_size),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
# }
#
# # Create training and validation datasets
# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
# # Create training and validation dataloaders
# dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}
# print(image_datasets)
#
# # Detect if we have a GPU available
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# def train_model(model, dataloaders, criterion, optimizer, num_epochs):
#     val_acc_history = []
#
#     return model
#
#
# def initialize_model(num_classes, feature_extract):
#     model = torch.hub.load('pytorch/vision:v0.5.0', 'deeplabv3_resnet101', pretrained=True)
#     # we only want to compute gradients for the newly initialized layer, not the others
#     if feature_extract:
#         for param in model.parameters():
#             param.requires_grad = False
#
#     num_ftrs = model.fc.in_features
#     model.fc = nn.Linear(num_ftrs, num_classes)
#     return model
#
#
# model = initialize_model(2, feature_extract)


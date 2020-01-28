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

# Parameters
num_class = 2  # FG vs BG
batch_size = 8
num_epochs = 2
feature_extract = True  # True: only update reshaped layer params, False: finetune whole model
input_size = 224


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


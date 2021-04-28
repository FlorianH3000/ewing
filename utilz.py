"""
utilz
"""

import numpy as np
from sklearn.metrics import f1_score
from PIL import Image
import os
import torch
from scipy.ndimage.interpolation import map_coordinates
from scipy import interpolate as ipol
import random
from torchvision import models
import torch.nn as nn
from pydicom import dcmread
import matplotlib.pyplot as plt
from skimage.transform import rescale, resize

#########################################################################
def img_display(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    return npimg


def dice_score(label, pred):    
    x = f1_score(label, pred, average='weighted')
    return x



def load_img(path_img):
    _, ext = os.path.splitext(path_img)
    if len(ext) > 6:
        x = dcmread(path_img).pixel_array.astype(np.float32)/255
        h, w  = x.shape 
        minimum = np.minimum(h, w)
        maximum = np.maximum(h, w)

        margin = int((maximum - minimum)/2)
        if h > w:
            x = x[margin:(h-margin),:]
        if h < w:
            x = x[:,margin:(w-margin)]
        x = x[:minimum, :minimum]
        # print(x.shape)
        x = resize(x, (25,25))
        x = np.dstack((x,x,x))
        x = np.transpose(x, (2,0,1))
 
    return x
        

        




def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        
    if model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        
    if model_name == "resnet152":
        """ Resnet152
        """
        model_ft = models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
       # input_size = 224


    elif model_name == "densenet121":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
   
   
    elif model_name == "vgg11":
        """ VGG11_bn
        """
        model_ft = torch.hub.load('pytorch/vision:v0.6.0', 'vgg11_bn', pretrained=True)
        model_ft.classifier[6] = nn.Linear(4096,num_classes)
        print(model_ft)  
   
    elif model_name == "vgg13":
        """ VGG13_bn
        """
        model_ft = torch.hub.load('pytorch/vision:v0.6.0', 'vgg13_bn', pretrained=True)
        model_ft.classifier[6] = nn.Linear(4096,num_classes)


    elif model_name == "vgg19":
        """ VGG19_bn
        """
        model_ft = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19_bn', pretrained=True)
        model_ft.classifier[6] = nn.Linear(4096,num_classes)

    else:
        print("Invalid model name, exiting...")

    return model_ft



def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


    

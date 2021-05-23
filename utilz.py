"""
utilz
"""

import numpy as np
from sklearn.metrics import f1_score
import os
import torch
import random
from torchvision import models
import torch.nn as nn
from pydicom import dcmread
import matplotlib.pyplot as plt
from skimage.transform import resize
from arguments import input_size

#########################################################################
# def img_display(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     npimg = np.transpose(npimg, (1, 2, 0))
#     return npimg


def dice_score(label, pred):    
    x = f1_score(label, pred, average='weighted')
    return x



def load_img(path_img):
    _, ext = os.path.splitext(path_img)
    if len(ext) > 6:
        x = dcmread(path_img).pixel_array.astype(np.float32)/255

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


    return model_ft



def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


    


# =============================================================================
# TRANSFORM
# =============================================================================

def transforms(img, phase):
    
    ### normalizing
    img = np.nan_to_num(img)
    img = img - img.mean()
    img = img / img.max()
    img = np.nan_to_num(img)
           
    # if phase == 'train':
        
    #     ###horizontal_flip
    #     if 0.5 > random.random():
    #         img = np.copy(np.flip(img, axis=1))
        
    #     #vertical_flip
    #     if 0 > random.random():
    #         img = np.copy(np.flip(img, axis=0))
         
    #     ### random_crop
    #     if 0.5 > random.random():
    #         h, w  = img.shape 
    #         minimum = np.minimum(h, w)
    #         margin = int(minimum*0.2)
    #         margin_final = int(random.randint(0, margin)/2)
    #         img = img[margin_final:(h-margin_final),margin_final:(w-margin_final)]


        
    ### validation and test only
    h, w  = img.shape 
    minimum = np.minimum(h, w)
    maximum = np.maximum(h, w)

    margin = int((maximum - minimum)/2)
    if h > w:
        img = img[margin:(h-margin),:]
    if h < w:
        img = img[:,margin:(w-margin)]
    img = img[:minimum, :minimum]
    img = resize(img, (input_size,input_size))
    img = np.dstack((img,img,img))
    img = np.transpose(img, (2,0,1))    
    img = torch.from_numpy(img)
    
    # print(img.shape)
    # plt.figure()
    # plt.imshow(img[0,:,:])
   
    return img



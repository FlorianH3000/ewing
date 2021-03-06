"""
utility functions
"""

import numpy as np
import torch
import random
from torchvision import models
import torch.nn as nn
from skimage.transform import resize
from arguments import args
from scipy import ndimage

#########################################################################



      
def calculate_sensitivity_specificity(y_test, y_pred_test, num_classes):
    true_pos, false_pos, true_neg, false_neg, acc = 0,0,0,0,0
    for (l,p) in zip(y_test, y_pred_test): 
        
        for (l_, p_) in zip(l, p):
            if l_ == p_ == 1:
                true_pos += 1
            if l_ == 0 and p_ == 1:
                false_pos += 1
            if l_ == p_ == 0:
                true_neg += 1
            if l_ == 1 and p_ == 0:
                false_neg += 1
            if l_ == p_:
                acc += 1
            
    if num_classes > 2:
        accuracy = acc / len(y_pred_test)
        return 0, 0, accuracy
        
    else:
        # Calculate accuracy
        accuracy = acc / (len(y_pred_test) * args['batch_size'])
        # Calculate sensitivity and specificity
        sensitivity = true_pos / (true_pos + false_neg)
        specificity = true_neg / (true_neg + false_pos)
        
        return sensitivity, specificity, accuracy  




def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):

    if model_name == "resnet18":
        """ Resnet18
        """
        model = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    if model_name == "resnet34":
        model = models.resnet34(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

        
    if model_name == "resnet50":
        model = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
    if model_name == "resnet152":
        model = models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == "alexnet":
        model = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs,num_classes)
        
    elif model_name == "densenet121":
        model = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
   
    elif model_name == "vgg11":
        model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg11_bn', pretrained=True)
        model.classifier[6] = nn.Linear(4096,num_classes)
   
    return model



def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


    

# =============================================================================
# Adjsut image to suitable size and format
# transforms
# =============================================================================

def adjust_img(img, phase):
    
    ### normalizing according to current data
    img = np.nan_to_num(img)
    img = img - img.mean()
    img = img / img.max()
    img = np.nan_to_num(img)
    
    if phase == 'train':
        
        if args['augm_hflip']:
            if 0.3 > random.random():
                img = np.copy(np.flip(img, axis=1))
        if args['augm_vflip']:
            if 0 > random.random():
                img = np.copy(np.flip(img, axis=0))
        if args['augm_crop']:
            if 0.3 > random.random():
                h, w, _  = img.shape 
                minimum = np.minimum(h, w)
                margin = int(minimum*0.1)
                margin_final = int(random.randint(0, margin)/2)
                img = img[margin_final:(h-margin_final),margin_final:(w-margin_final)]
        if args['augm_rotate']:
            if 0.3 > random.random():
                x = random.randint(-25, 25)
                img = ndimage.rotate(img, x, reshape=False)
    
    h, w, _  = img.shape 
    minimum = np.minimum(h, w)
    maximum = np.maximum(h, w)

    margin = int((maximum - minimum)/2)
    if h > w:
        img = img[margin:(h-margin),:]
    if h < w:
        img = img[:,margin:(w-margin)]
    img = img[:minimum, :minimum]
    img = resize(img, (args['input_size'], args['input_size']))
    # img = np.dstack((img,img, img))
    img = np.transpose(img, (2,0,1))    
    img = torch.from_numpy(img)
       
    return img

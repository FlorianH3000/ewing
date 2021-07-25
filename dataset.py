"""
ImgDataset
"""

from torch.utils.data import Dataset
import os
import utilz as ut
import matplotlib.pyplot as plt
import numpy as np
import cv2

class ImgDataset(Dataset):

    def __init__(self, img, lab, phase, transf):

        self.img = img
        self.lab = lab
        self.phase = phase
        self.transf = transf


    def __len__(self):
        return len(self.img)
    
    def __getitem__(self, idx):
     
        # img = plt.imread(self.img[idx])
        img = cv2.imread(self.img[idx])
        img = ut.adjust_img(img, self.phase).float()
        img = self.transf(img)

        
            
        lab = self.lab[idx]

        sample = {'img': img, 'lab': lab}      
        
        return sample
    
    
    
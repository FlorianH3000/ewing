"""
ImgDataset
"""

from torch.utils.data import Dataset
import os
from utilz import load_img
import matplotlib.pyplot as plt
import numpy as np


class ImgDataset(Dataset):

    def __init__(self, img, lab, transf):

        self.img = img
        self.lab = lab
        self.transf = transf


    def __len__(self):
        return len(self.img)
    
    def __getitem__(self, idx):
     
        img = load_img(self.img[idx])

        lab = self.lab[idx]
        
        # img = self.transf(img)
        
        sample = {'img': img, 'lab': lab}      
        
        return sample
    
    
    
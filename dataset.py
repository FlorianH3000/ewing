"""
ImgDataset
"""

from torch.utils.data import Dataset
import os
import utilz as ut
import matplotlib.pyplot as plt
import numpy as np


class ImgDataset(Dataset):

    def __init__(self, img, lab, phase):

        self.img = img
        self.lab = lab
        self.phase = phase


    def __len__(self):
        return len(self.img)
    
    def __getitem__(self, idx):
     
        img = ut.load_img(self.img[idx])
        img = ut.transforms(img, self.phase)

        lab = self.lab[idx]

        sample = {'img': img, 'lab': lab}      
        
        return sample
    
    
    
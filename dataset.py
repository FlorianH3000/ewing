"""
ImgDataset
"""

from torch.utils.data import Dataset
import os
from utilz import load_img
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

class ImgDataset(Dataset):

    def __init__(self, list_, transf):

        self.list_ = list_
        self.transf = transf


    def __len__(self):
        return len(self.list_)
    
    def __getitem__(self, idx):
     
        img = load_img(self.list_[idx])
        
        # img = self.transf(img)

        
        x = os.path.dirname(self.list_[idx])
        _,lab = os.path.split(x)

        lab = int(lab[0])

        
        sample = {'img': img, 'lab': lab}      


        return sample
    
    
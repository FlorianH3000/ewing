"""
custom ImgDataset
"""

from torch.utils.data import Dataset
import utilz as ut
import cv2

class ImgDataset(Dataset):

    def __init__(self, img, lab, phase):

        self.img = img
        self.lab = lab
        self.phase = phase


    def __len__(self):
        return len(self.img)
    
    def __getitem__(self, idx):
     
        img = cv2.imread(self.img[idx])
        img = ut.adjust_img(img, self.phase).float()
        
            
        lab = self.lab[idx]

        sample = {'img': img, 'lab': lab}      
        
        return sample
    
    
    
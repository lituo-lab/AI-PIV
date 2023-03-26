import os
import cv2
import torch
import struct
import numpy as np
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, root):
        self.dataset = []

        for filename in os.listdir(root):
            if filename[-1] == 'o':
                self.dataset.append(os.path.join(root, filename))

    def __len__(self):
        return len(self.dataset)
    

    def __getitem__(self, index):
        
        flow = self.dataset[index]
        flows = self.GetVelocity(flow)
        
        imgOne = flow.replace("flow.flo", "img1.tif")
        tenOne = torch.tensor(cv2.imread(imgOne,cv2.IMREAD_GRAYSCALE)).float()/255
        
        imgTwo = flow.replace("flow.flo", "img2.tif")
        tenTwo = torch.tensor(cv2.imread(imgTwo,cv2.IMREAD_GRAYSCALE)).float()/255
        
        images = torch.stack([tenOne,tenTwo], dim=0)
        
        return images, flows
        
    @staticmethod
    def GetVelocity(filename):
        fid = open(filename, 'rb')
        tag = struct.unpack('f',fid.read(4))
        width = struct.unpack('i',fid.read(4))
        height = struct.unpack('i',fid.read(4))
        flow = struct.unpack(256*256*2*'f',fid.read(-1))
        flow = np.array(flow,dtype=float).reshape((256, 256*2))
        
        u = flow[:, np.arange(0,256*2,2)]; 
        v = flow[:, np.arange(1,256*2,2)]; 
        fid.close()
        
        u = torch.tensor(u).float()
        v = torch.tensor(v).float()
        
        return torch.stack([u,v], dim=0)


if __name__ == '__main__':
    data = MyDataset('.\DNS_turbulence')

    for i in range(1, 2):
        images, flows = data[i]
        
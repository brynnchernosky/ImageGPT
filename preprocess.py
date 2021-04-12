from torch.utils.data import Dataset, DataLoader
import numpy as np
from skimage.io import imread
import os
from PIL import Image
import pickle

class ImageDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        color_images = dict["data"] #10000x3072, "Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image."

        #coefficients for converting to grayscale from LUMA-REC.709
        red_channels = color_images[:,:1024]*0.2125
        green_channels = color_images[:,1024:2048]*0.7154
        blue_channels = color_images[:,2048:]*0.0721
        self.data = red_channels+green_channels+blue_channels #10000x1024
        print(self.data[0,:50])

    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = {
            "input": self.data[idx]
        }
        return item
    

def load_dataset(file, batch_size):
    data = ImageDataset(file)
    loader = DataLoader(data,batch_size,shuffle=True)
    return loader
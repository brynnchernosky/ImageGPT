from torch.utils.data import Dataset, DataLoader
import numpy as np
from skimage.io import imread
import os
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path

        file_list = []
        for root, dirs, files in os.walk(self.data_path):
            for name in files:
                if name.endswith(".jpg") or name.endswith(".png"):
                    file_list.append(os.path.join(root, name))
        
        self.data = np.zeros((len(file_list), 32, 32, 3))

        for i, file_path in enumerate(file_list):
            img = Image.open(file_path)
            img = img.resize((32, 32))
            img = np.array(img, dtype=np.float32)
            img /= 255.

            # Converts Grayscale images to RGB through stacking if they're currently not
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=-1)

            self.data[i] = img
    
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
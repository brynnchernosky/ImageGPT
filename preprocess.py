import numpy as np
from skimage.io import imread
import os
from PIL import Image


class Datasets():

    def __init__(self, data_path, task):
        self.data_path = data_path
        self.task = task

        self.data = np.zeros(
            (200, 32, 32, 3))

        file_list = []
        for root, dirs, files in os.walk(self.data_path):
            for name in files:
                if name.endswith(".jpg") or name.endswith(".png"):
                    file_list.append(os.path.join(root, name))

        for i, file_path in enumerate(file_list):
            img = Image.open(file_path)
            img = img.resize((32, 32))
            img = np.array(img, dtype=np.float32)
            img /= 255.

            # Converts Grayscale images to RGB through stacking if they're currently not
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=-1)

            self.data[i] = img
    
    def load_dataset(self, train, test):
        self.inputs = self.data
        self.labels = np.split(self.data, 2)[0]
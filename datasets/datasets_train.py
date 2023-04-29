import cv2
import os
import torch # 
import numpy as np 
from torch.utils import data
from torchvision import transforms

class DatasetDogCat(data.Dataset) :
    def __init__(self, path_data, load_width, load_height, nb_classes) -> None:
        super(DatasetDogCat, self).__init__()
        self.load_width = load_width
        self.load_height = load_height # 480*360*3 = 224*224*3
        self.nb_classes = nb_classes # numbers classes = 2 cho meo
        self.path_data = path_data
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.load_height, self.load_width)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) #RBG - RGB 0-1 tensor type numpy int float gpu để train
        ])
        path_images = [] # đường dẫn  data/train/Cat/0.jpg 1 data/train/dog/
        labels = [] # 0 1 0 1 0 1 0 1 1 1 1  0 0 0 1 0 1 0 1 0 0 
        dict_class = {'Dog': 1, 'Cat' : 0}

        for label in os.listdir(self.path_data) : # data/train = ['Cat', 'Dog'] 'data', 'train' = data/train
            for fname in os.listdir(os.path.join(self.path_data, label)) : #os.path.join(self.path_data, label)  =  data/train/Cat
                if '.jpg' in fname or '.png' in fname :
                    path_images.append(os.path.join(self.path_data, label, fname))
                    labels.append(dict_class[label]) # cho meo, 1 0
        
        self.path_images = path_images
        self.labels = labels
    def __getitem__(self, index) :
        path_image = self.path_images[index]
        label = self.labels[index]
        image = cv2.imread(path_image) # đọc ảnh return 480*360*3 0-255
        image = self.transform(image) # 0-1 
        result = {
            'path_image' : path_image,
            'image' : image,
            'label' : label
        }
        return result
    
    def __len__(self):
        return len(self.labels)

    
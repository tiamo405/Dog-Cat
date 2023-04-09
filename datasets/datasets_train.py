import cv2
import os
import torch
import numpy as np 
from torch.utils import data
from torchvision import transforms
class DatasetDogCat(data.Dataset) :
    def __init__(self, path_data, load_width, load_height, nb_classes) -> None:
        super(DatasetDogCat, self).__init__()
        self.load_width = load_width
        self.load_height = load_height
        self.nb_classes = nb_classes
        self.path_data = path_data
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            # transforms.CenterCrop(0.5),
            transforms.Resize((self.load_height, self.load_width)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        path_images = []
        labels = []
        dict_class = {'Dog': 1, 'Cat' : 0}

        for label in os.listdir(self.path_data) :
            for fname in os.listdir(os.path.join(self.path_data, label)) :
                path_images.append(os.path.join(self.path_data, label, fname))
                labels.append(dict_class[label])
        
        self.path_images = path_images
        self.labels = labels
    def __getitem__(self, index) :
        path_image = self.path_images[index]
        label = self.labels[index]
        image = cv2.imread(path_image)
        image = self.transform(image)
        result = {
            'path_image' : path_image,
            'image' : image,
            'label' : label
        }
        return result
    
    def __len__(self):
        return len(self.labels)

    
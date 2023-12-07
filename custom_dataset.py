import torchvision
import torchvision.transforms as transforms
import random
import torch
from PIL import Image
import numpy
import math
import matplotlib.pyplot as plt



class CustomDataset():
    
    def __init__(self, C = 10):
        
        
        self.ds = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform = None)
        
        self.C = C
        


    def __getitem__(self, index):
        
        image, label = self.ds[index]
        
        is_flipped = random.randint(0, 1)
        
       
        if is_flipped:
            
            image = image.transpose(Image.ROTATE_180)
        
        
        
        if is_flipped:
          
            label = label + 10
        
        image = transforms.ToTensor()(image) 
    


        return image, label
    
    
    
    def show(self, image):
        
        image = transforms.ToPILImage()(image).convert("RGB")
        
        fig, ax = plt.subplots()
        
        ax.imshow(image)
        
        plt.show()
        

    def __len__(self):
        
        return len(self.ds)
   
    
    def yolo_collate_fn(self, batch):
        
        images = []
        
        labels = []

        for sample in batch:
            
            image, label = sample
            
            images.append(image)
            
            labels.append(label)

        
        images = torch.stack(images)
        
        targets = torch.stack(labels)

        return images, labels
    


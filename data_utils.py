import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms 
from datasets import load_dataset

class SmithsonianButterflies(Dataset):
    def __init__(self, image_size=128, context=False):
        self.dataset = load_dataset('huggan/smithsonian_butterflies_subset', split="train", cache_dir='cache_dir')
        self.image_size = image_size
        self.images = [sample['image'] for sample in self.dataset]
        self.class_names = [sample['name'] for sample in self.dataset]

        self.num_classes = 0
        self.class_mapping = {}
        for class_name in self.class_names:
            if class_name not in self.class_mapping:
                self.class_mapping[class_name] = self.num_classes
                self.num_classes += 1
    
        self.labels = [self.class_mapping[name] for name in self.class_names]

        self.preprocess = transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
            
    def transform(self, image):
        image = image.convert('RGB')
        image = self.preprocess(image)
        return image

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return self.transform(image), label



class SpriteDataset(Dataset):
    def __init__(self, sfilename, lfilename, null_context=False):
        self.sprites = np.load(sfilename)
        self.slabels = np.load(lfilename)
        # print(f"sprite shape: {self.sprites.shape}")
        # print(f"labels shape: {self.slabels.shape}")
        self.null_context = null_context
        self.sprites_shape = self.sprites.shape
        self.slabel_shape = self.slabels.shape
        self.transform = transforms.Compose([
                transforms.ToTensor(),                # from [0,255] to range [0.0,1.0]
                transforms.Normalize((0.5,), (0.5,))  # range [-1,1]
            ])
                
    # Return the number of images in the dataset
    def __len__(self):
        return len(self.sprites)
    
    # Get the image and label at a given index
    def __getitem__(self, idx):
        # Return the image and label as a tuple
        if self.transform:
            image = self.transform(self.sprites[idx])
            if self.null_context:
                label = torch.tensor(0).to(torch.int64)
            else:
                label = torch.tensor(self.slabels[idx]).to(torch.int64)
        return (image, label)

    def getshapes(self):
        # return shapes of data and labels
        return self.sprites_shape, self.slabel_shape


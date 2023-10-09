from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
import torch
from torchvision import transforms as T


class ChalearnDataset(Dataset):
    def __init__(self, data_df, root_dir, transforms):
        self.data = data_df
        self.transforms = transforms
        self.root_dir = root_dir
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        image = Image.open(os.path.join(self.root_dir, sample["file_path"]))
        image = self.transforms(image)
        labels = torch.Tensor(list(sample[1:-1].values))
        return image, labels


class ChalearnInferDataset(Dataset):
    def __init__(self, data_df, root_dir, transforms):
        self.data = data_df
        self.transforms = transforms
        self.root_dir = root_dir
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        image = Image.open(os.path.join(self.root_dir, sample["file_path"]))
        image = self.transforms(image)
        return image


def build_dataloader(train_df, val_df, root_dir, batch_size):
    train_transforms = T.Compose([
        T.RandomResizedCrop(size=(224, 224)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = ChalearnDataset(train_df, root_dir, train_transforms)
    val_dataset = ChalearnDataset(val_df, root_dir, test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader

def build_test_loader(test_df, root_dir, test_batch_size):
    test_transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = ChalearnInferDataset(test_df, root_dir, test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, drop_last=False)

    return test_loader
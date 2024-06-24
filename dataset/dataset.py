from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch
from torchvision import transforms as T
import scipy.io 
from PIL import Image
import numpy as np
from dataset.pedes import PedesAttrPETA

pa_100k_group_order = [7,8,13,14,15,16,17,18,19,20,21,22,23,24,25,9,10,11,12,1,2,3,0,4,5,6]
pa_100k_num_in_group = [2, 6, 6, 1, 4, 3, 1, 3]


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

class PA100KDataset(Dataset):
    def __init__(self, root_dir, transforms, split, use_multitask=False):
        self.annotations = scipy.io.loadmat("./data/PA-100K/annotation.mat")
        self.file_paths = self.annotations[f"{split}_images_name"]
        self.labels = self.annotations[f"{split}_label"]
        self.root_dir = root_dir
        self.transforms = transforms
        self.use_multitask = use_multitask
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image_path = self.file_paths[index][0][0]
        image = Image.open(os.path.join(self.root_dir, image_path))
        if self.transforms:
            image = self.transforms(image)
        if self.use_multitask:
            group_label = []
            label = self.labels[index]

            for group in range(len(pa_100k_num_in_group)):
                group_num = pa_100k_num_in_group[group]
                start_index = pa_100k_group_order[sum(pa_100k_num_in_group[:group])]
                end_index = pa_100k_group_order[sum(pa_100k_num_in_group[:group]) + group_num - 1]
                group_label.append(np.argmax(self.labels[index][start_index:end_index+1]))
            return image, group_label

        label = self.labels[index]
        return image, label
        

def build_dataloader(cfg, image_res, root_dir, batch_size, train_df=None, val_df=None, data_name="PA100K", use_multi_task=False):
    
    train_transforms = T.Compose([
        T.Resize(image_res),
        T.Pad(10),
        T.RandomCrop(image_res),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transforms = T.Compose([
        T.Resize(image_res),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if data_name == "upar":
        train_dataset = ChalearnDataset(train_df, root_dir, train_transforms)
        val_dataset = ChalearnDataset(val_df, root_dir, test_transforms)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    elif data_name == "PA100K":
        train_dataset = PA100KDataset(root_dir, train_transforms, "train", use_multitask=use_multi_task)
        val_dataset = PA100KDataset(root_dir, test_transforms, "test", use_multitask=use_multi_task)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    elif data_name == "PETA":
        train_dataset = PedesAttrPETA(cfg, split="trainval", transform=train_transforms, target_transform=[])
        val_dataset = PedesAttrPETA(cfg, split="test", transform=test_transforms, target_transform=[])
    else:
        raise NotImplementedError("Invalid dataset name!")
    
    return train_loader, val_loader, train_dataset, val_dataset

def build_test_loader(image_res, root_dir, test_batch_size, test_df=None, data_name="PA100K"):
    test_transforms = T.Compose([
        T.Resize(image_res),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if data_name == "upar":
        test_dataset = ChalearnInferDataset(test_df, root_dir, test_transforms)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, drop_last=False)
    elif data_name == "PA100K":
        test_dataset = PA100KDataset(root_dir, test_transforms, "test")
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, drop_last=False)

    return test_loader
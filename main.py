from sklearn.model_selection import train_test_split
import yaml
import pandas as pd
import numpy as np
import torch
import random
from dataset.dataset import build_dataloader
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from models import PARModel
import torch.optim as optim
import torch.nn as nn
from train import train
import argparse
from losses import build_loss

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def main(config):
    device = torch.device("cuda")
    
    if config["data_name"] == "upar":
        data_df = pd.read_csv(config["data_path"]).reset_index().rename(columns={"index": "file_path"})
        set_seed(config["seed"])
        data_df["data_source"] = data_df["file_path"].map(lambda x: x.split("/")[0])

        train_df, val_df = train_test_split(data_df, test_size=0.2, random_state=config["seed"], stratify=data_df["data_source"])
        train_loader, val_loader = build_dataloader(cfg=config, train_df=train_df, val_df=val_df, root_dir=config["root_dir"], 
                                                    batch_size=config["train_batch_size"], data_name=config["data_name"])
    elif config["data_name"] in ["PA100K", "PETA"]:
        if isinstance(config["image_res"], list):
            image_res = tuple(config["image_res"])
            config["image_res"] = image_res
        else:
            image_res = config["image_res"]
        train_loader, val_loader, train_set, val_set = build_dataloader(
            cfg=config,
            image_res=image_res,
            root_dir=config["root_dir"], 
            batch_size=config["train_batch_size"], 
            data_name=config["data_name"],
            use_multi_task=config["use_multi_task"])
    else:
        raise NotImplementedError("Invalid dataset name!")
    
    model = PARModel(config)
    
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])

    labels = train_set.labels
    label_ratio = labels.mean(0) if config["sample_weight"] else None
    criterion = build_loss(config["loss"], sample_weight=label_ratio, scale=config["scale"], size_sum=True)
    criterion = criterion.cuda()

    scaler = torch.cuda.amp.GradScaler()
    
    if config["scheduler"] == "cosine": 
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["num_epochs"] * len(train_loader))
    elif config["scheduler"] == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=4)
    else:
        raise NotImplementedError(f"learning rate scheduler {config["scheduler"]} not implemented!")
    train(config, model, train_loader, val_loader, optimizer, criterion, device=device, scaler=scaler, scheduler=scheduler)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/baseline_pa100k.yaml", type=str, help="Path to the YAML file containing the configuration")

    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    main(config)
from sklearn.model_selection import train_test_split
import yaml
import pandas as pd
import numpy as np
import torch
import random
from dataset import build_dataloader
from model import ChalearnModel
import torch.optim as optim
import torch.nn as nn
from train import train
import argparse

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def main(config):
    device = torch.device("cuda")
    
    data_df = pd.read_csv(config["data_path"]).reset_index().rename(columns={"index": "file_path"})
    set_seed(config["seed"])
    data_df["data_source"] = data_df["file_path"].map(lambda x: x.split("/")[0])

    train_df, val_df = train_test_split(data_df, test_size=0.2, random_state=config["seed"], stratify=data_df["data_source"])
    train_loader, val_loader = build_dataloader(train_df, val_df, config["root_dir"], config["train_batch_size"])
    model = ChalearnModel(num_attributes=40, backbone_name='convnext_small', pretrained=True)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    
    train(config, model, train_loader, val_loader, optimizer, criterion, device=device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the YAML file containing the configuration")

    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    main(config)
from dataset import build_test_loader
import torch
from tqdm import tqdm
import os
import pandas as pd
from model import ChalearnModel

def test(model, test_loader, test_df, pred_columns, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), test_bs=96):
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        for idx, inputs in tqdm(enumerate(test_loader), total=len(test_loader)):
            # Forward pass
            outputs = model(inputs.to(device))
            # Collect predictions and true targets
            #for i in range(len(outputs)):
            pred = (outputs.cpu().numpy() > 0.5).astype(int)
            test_df.loc[test_bs * idx: test_bs * idx + pred.shape[0] - 1, pred_columns] = pred
                
    return test_df

if __name__ == "__main__":
    test_df = pd.read_csv("./data/annotations/phase1/val_task1/val.csv", header=None).rename(columns={0: "file_path"})
    data_df = pd.read_csv("./data/annotations/phase1/train/train.csv").reset_index().rename(columns={"index": "file_path"})
    pred_columns = data_df.columns[1:]

    model = ChalearnModel(num_attributes=40, backbone_name='convnext_small', pretrained=True)
    test_loader = build_test_loader(test_df, "./data", 96)

    model.load_state_dict(torch.load("./checkpoint/model.pth", map_location="cpu"))
    test_df = test(model, test_loader, test_df, pred_columns)
    test_df[pred_columns] = test_df[pred_columns].astype(int)
    test_df.to_csv("./output/predictions.csv", index=False)

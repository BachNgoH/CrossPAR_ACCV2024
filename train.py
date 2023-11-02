import torch
from tqdm import tqdm
import numpy as np
from metrics import get_pedestrian_metrics

def evaluate(cfg, model, val_loader, device, epoch=0, best_acc=None):
        # Validation
    model.eval()  # Set the model to evaluation mode

    preds_probs = []
    gt_list = []

    with torch.no_grad():
        for (inputs, gt_label) in val_loader:
            # Forward pass
            gt_label = gt_label.cuda()
            gt_list.append(gt_label.cpu().numpy())
            
            outputs = model(inputs.to(device))
            probs = outputs.sigmoid()
            preds_probs.append(probs.cpu().numpy())
            
        gt_label = np.concatenate(gt_list, axis=0)
        preds_probs = np.concatenate(preds_probs, axis=0)

        val_results = get_pedestrian_metrics(gt_label, preds_probs)
        mean_results = (val_results.ma + val_results.instance_acc + val_results.instance_prec + val_results.instance_recall + val_results.instance_f1) / 5
        
        print(f'Epoch [{epoch+1}/{cfg["num_epochs"]}] - ' \
            f'mA: {val_results.ma:.4f} - InsAcc: {val_results.instance_acc} - InsPrec: {val_results.instance_prec} - ' \
            f'InsRecall: {val_results.instance_recall} - InsF1: {val_results.instance_f1} - mean results: {mean_results}')
        

        if best_acc and mean_results > best_acc:
            # Save the trained model
            best_acc = mean_results
            torch.save(model.state_dict(), './checkpoint/model_best.pth')

def train(cfg, model, train_loader, val_loader, optimizer, criterion, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
          scaler=None, scheduler=None):
    # Training loop
    print("Using device:", device)
    best_acc = 0.0
    for epoch in range(cfg["num_epochs"]):
        model.to(device)
        model.train()  # Set the model to training mode
        total_loss = 0.0

        for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader), total=len(train_loader)):
            
            # Zero the gradients
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                # Forward pass
                outputs = model(inputs.to(device))
                # Calculate loss
                loss = criterion(outputs, targets.to(device).float())[0][0]
            
            # Backpropagation and optimization
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            scheduler.step()
        print(f'Epoch [{epoch + 1}/{cfg["num_epochs"]}] Loss: {total_loss / (batch_idx + 1)}')


        evaluate(cfg, model, val_loader, device, epoch, best_acc)

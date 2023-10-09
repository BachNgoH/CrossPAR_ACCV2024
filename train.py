import torch
from tqdm import tqdm
import numpy as np


def evaluate(cfg, model, val_loader, device, epoch, best_acc):
        # Validation
    model.eval()  # Set the model to evaluation mode
    
    accuracies = []

    with torch.no_grad():
        for (inputs, targets) in val_loader:
            # Forward pass
            outputs = model(inputs.to(device))
            pred = (outputs > 0.5).float()
            correct = (pred == targets.to(device)).float()
            acc = torch.mean(correct).cpu()
            accuracies.append(acc)
            
        
        print(f'Epoch [{epoch+1}/{cfg["num_epochs"]}] - Validation Accuracy: {np.array(accuracies).mean():.4f}')
        if np.array(accuracies).mean() > best_acc:
            # Save the trained model
            best_acc = np.array(accuracies).mean()
            torch.save(model.state_dict(), 'model.pth')

def train(cfg, model, train_loader, val_loader, optimizer, criterion, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    # Training loop
    print("Using device:", device)
    best_acc = 0.0
    for epoch in range(cfg["num_epochs"]):
        model.to(device)
        model.train()  # Set the model to training mode
        
        for (inputs, targets) in tqdm(train_loader, total=len(train_loader)):
            
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs.to(device))

            # Calculate loss
            loss = criterion(outputs, targets.to(device))
            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

        evaluate(cfg, model, val_loader, device, epoch, best_acc)

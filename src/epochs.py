import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def train_epoch(model, loader, optimizer, loss_fn, device='cuda'):
    model.train()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        all_predictions.extend(output.argmax(1).cpu().detach().numpy())
        all_targets.extend(target.cpu().detach().numpy())
    
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    
    return avg_loss, accuracy, f1

def evaluate_epoch(model, loader, loss_fn, device='cuda'):

    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = loss_fn(output, target)
            
            total_loss += loss.item()
            
            all_predictions.extend(output.argmax(1).cpu().detach().numpy())
            all_targets.extend(target.cpu().detach().numpy())
    
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    
    return avg_loss, accuracy, f1
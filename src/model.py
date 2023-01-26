import torch
from torch import Tensor
from torch.nn import Linear, Parameter
import torch.nn.functional as F

from torch_geometric.nn import (GCNConv, MessagePassing, GATConv, GATv2Conv,GINConv,BatchNorm,
                                global_mean_pool, global_max_pool, global_add_pool)
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.loader import DataLoader

import numpy as np
from sklearn.metrics import f1_score, precision_score, accuracy_score
from sklearn.utils import class_weight
from collections import Counter

from torch_geometric.nn.models import GIN,GCN,GAT

from torch_geometric.datasets import LRGBDataset
dataset = LRGBDataset.LRGBDataset(root='/tmp/lrgb', name='PascalVOC-SP')

class GATv2(torch.nn.Module):
    def __init__(self, hidden_channels, heads):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.heads = heads
        
        self.conv1 = GATv2Conv(dataset.num_features, hidden_channels, heads = heads, dropout=0.6)
        self.conv2 = GATv2Conv(self.hidden_channels * self.heads, dataset.num_classes, concat=False, heads=1, dropout=0.6)
        
    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.0, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.0, training=self.training)
        x = self.conv2(x, edge_index)
        return x



def model_list():
    
    model_list = [GIN(in_channels = dataset.num_features,  hidden_channels=21, num_layers=3),
             GAT(in_channels=dataset.num_features, hidden_channels=220, num_layers=2), 
             GCN(in_channels = dataset.num_features, hidden_channels = 220, num_layers=8)
             ]
    
    return model_list


def run(model,max_epoch=500):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)

    print('\n',model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0)
    criterion = torch.nn.CrossEntropyLoss()
    #criterion = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(cw).float().to(device),reduction='mean')
    losses = []

    def train():
        for data in loader:
            data = data.to(device)
            model.train()
            optimizer.zero_grad()  # Clear gradients.
            out = model(data.x, data.edge_index)  # Perform a single forward pass.
            loss = criterion(out, data.y)  # Compute the loss solely based on the training nodes.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            return loss

    def val(mask):
        for data in loader:
            data = data.to(device)
            model.eval()
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct = pred == data.y  # Check against ground-truth labels.
            #acc = int(torch.sum(correct)) / int(torch.sum(correct))  # Derive ratio of correct predictions.
            acc = f1_score(data.y.cpu(),pred.cpu(),average='macro')
            #acc = accuracy_score(data.y.cpu(),pred.cpu())
            return acc


    for epoch in range(1, max_epoch):
        loss = train()
        val_acc = val(dataset)
        
        losses.append(loss.cpu().item())
        
    #   if epoch>5:
    #        if prev-test_acc<0.0001:
    #            print('break at: ',epoch)
    #            break
    #    prev = test_acc
        if epoch<30:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Valiation metric: {val_acc:.4f}')

        if epoch>=(max_epoch-50):
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Valiation metric: {val_acc:.4f}')
        #print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')*2
    
    return losses
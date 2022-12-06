import torch
from torch import Tensor
from torch.nn import Linear, Parameter
import torch.nn.functional as F

from torch_geometric.nn import (GCNConv, MessagePassing, GATConv, GATv2Conv,GINConv,BatchNorm,
                                global_mean_pool, global_max_pool, global_add_pool)
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.loader import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, accuracy_score
from sklearn.utils import class_weight
from collections import Counter

class MLP(torch.nn.Module):
    '''
    Simple Multilayer Perceptron for Graph-level Regression/Classification
    '''
    def __init__(self,in_channels, hidden_channels):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        
        self.layers = torch.nn.Sequential(
          torch.nn.Flatten(),
          torch.nn.Linear(self.in_channels, self.hidden_channels),
          torch.nn.ReLU(),
          torch.nn.Linear(self.hidden_channels, 64),
          torch.nn.ReLU(),
          torch.nn.Linear(64, 10)
        )


    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)
    
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = MLP(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, heads):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.heads = heads
        
        self.conv1 = GATConv(dataset.num_features, hidden_channels, heads = heads, dropout=0.6)
        self.conv2 = GATConv(self.hidden_channels * self.heads, dataset.num_classes, concat=False, heads=1, dropout=0.6)
        self.lin = MLP(self.hidden_channels, dataset.num_classes)
        
    def forward(self, x, edge_index, batch):
        x = F.dropout(x, p=0.0, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.0, training=self.training)
        x = self.conv2(x, edge_index)

        x = global_mean_pool(x, batch)

        # 3. Apply a final classifier
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x
    
class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP(in_channels, hidden_channels)
            self.convs.append(GINConv(nn=mlp, train_eps=False))
            in_channels = hidden_channels

        self.mlp = MLP(in_channels, hidden_channels)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = global_add_pool(x, batch)
        return self.mlp(x)

    
def get_models():
    return [GCN(hidden_channels=208),model = GAT(hidden_channels=10,heads=12), Net(dataset.num_features, 10, 10, 2)]

def run(model):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=0.0)
    criterion = torch.nn.CrossEntropyLoss()
    #criterion = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(cw).float().to(device),reduction='mean')

    def train():
        for data in loader:
            data = data.to(device)
            model.train()
            optimizer.zero_grad()  # Clear gradients.
            out = model(data.x.float(), data.edge_index,data.batch)  # Perform a single forward pass.
            loss = criterion(out, data.y)  # Compute the loss solely based on the training nodes.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            return loss

    def test(mask):
        for data in loader:
            data = data.to(device)
            model.eval()
            out = model(data.x.float(), data.edge_index,data.batch)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            y = data.y.argmax(dim=1)
            acc = precision_score(y.cpu(),pred.cpu(),average='micro')
            return acc
    
    losses = []

    for epoch in range(1, 100):
        loss = train()
    #    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
        #val_acc = test(data.val_mask)
        test_acc = test(dataset)
        losses.append(loss.cpu().item())

    #    if epoch>5:
    #        if prev-test_acc<0.0001:
    #            print('break at: ',epoch)
    #            break

        #prev = test_acc

    #    if epoch>=450:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test: {test_acc:.4f}')
        #print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')*2
    return losses
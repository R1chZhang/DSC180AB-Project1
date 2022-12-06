import torch
from typing import Callable, Optional
from torch_geometric.data import Data, InMemoryDataset

class myTestData(InMemoryDataset):
    """
    test data generation
    """
    def __init__(self, transform: Optional[Callable] = None):
        super().__init__('.', transform)
              
        row = [ 7,  8, 12, 12,  6,  8, 12,  3,  8, 13,  7, 14,  6, 11,  1,  3, 11,
        3,  1,  1,  4, 10,  8,  3,  7,  2, 12,  9,  8,  5,  5,  4,  1,  9,
        4,  2,  7,  9,  5, 10,  7,  5,  5,  7,  2, 12,  7,  7,  6,  7, 13,
        5, 12, 14, 10, 10, 11,  9, 14, 14,  8,  8, 12, 12,  3, 11, 11,  9,
        7,  9,  8,  9,  5, 14,  6,  7, 11, 10,  2,  4,  6,  8, 10, 10, 10,
       14, 10,  6,  1,  7,  2, 12, 14,  4, 12,  9, 13, 14,  8, 11, 14,  2,
       12,  8,  1,  3,  5, 12, 10, 13,  2,  5,  1, 12,  1, 14,  9,  8, 12,
       11, 11, 11, 10, 13,  7,  3,  9,  2,  5,  2,  6,  8,  2,  7, 12,  8,
        5, 13,  4,  8, 12,  7,  8, 12,  2,  7, 12,  1,  5,  5]
        
        col = [ 2,  3, 13,  6,  8,  1,  7,  9, 11,  3, 12,  4, 14,  1,  4,  5,  4,
        3, 10,  3,  8, 11,  9, 10, 12,  6, 12,  5, 13,  1,  2,  5,  7, 12,
        7,  7,  7, 12,  6,  3,  5, 10,  7,  6,  1,  2,  6,  6, 14,  6,  4,
        4, 13, 11,  3, 14, 11,  2,  8, 10,  2,  5,  3,  9,  1, 14, 11, 12,
        4,  5,  6, 11,  5,  6,  2,  8,  8,  4,  3,  2,  2,  4,  3, 14, 11,
        4,  3,  8, 10,  7,  1,  7, 13, 12,  6, 12,  7,  6,  2,  6,  3, 12,
       14, 10, 14,  1,  4, 12, 11,  3,  9,  1,  1,  5,  2,  1,  9,  9, 13,
       11,  5, 10,  9, 11, 11, 14,  1, 13, 11, 12, 10, 14, 12, 14, 12,  8,
        3,  9,  8, 12,  8, 12,  1,  4,  3,  7, 14,  7, 14, 13]
        
        edge_index = torch.tensor([row, col])
        num_features = 34
        num_classes=4
        
        y = torch.tensor([  # Create communities.
            1, 3, 1, 0, 3, 1, 2, 0, 3, 0, 2, 0, 2, 1, 3, 0, 3, 1, 0, 1, 0, 1,
       3, 3, 0, 2, 3, 3, 2, 0, 0, 2, 3, 3])

        x = torch.eye(y.size(0), dtype=torch.float)

        train_mask = torch.zeros(y.size(0), dtype=torch.bool)
        for i in range(int(y.max()) + 1):
            train_mask[(y == i).nonzero(as_tuple=False)[0]] = True
        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                    num_features = num_features, num_classes=num_classes)
    
        self.data, self.slices = self.collate([data])
        
#data = myTestData().data
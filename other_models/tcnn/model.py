import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data as PyGData
class TCNN(nn.Module):
    """
    PyTorch implementation of tCNN <https://github.com/Lowpassfilter/tCNNS-Project>
    with default settings from the original
    """
    def __init__(self):
        super().__init__()
        self.drug_conv = nn.Sequential(
            nn.Conv1d(28,40,7),
            nn.ReLU(),
            nn.MaxPool1d(3,3),
            nn.Conv1d(40,80,7),
            nn.ReLU(),
            nn.MaxPool1d(3,3),
            nn.Conv1d(80,60,7),
            nn.ReLU(),
            nn.MaxPool1d(3,3),
        )
        self.cell_conv = nn.Sequential(
            nn.Conv1d(1,40,7),
            nn.ReLU(),
            nn.MaxPool1d(3,3),
            nn.Conv1d(40,80,7),
            nn.ReLU(),
            nn.MaxPool1d(3,3),
            nn.Conv1d(80,60,7),
            nn.ReLU(),
            nn.MaxPool1d(3,3),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(1024),
            nn.Dropout(),
            nn.Linear(1024,1024),
            nn.Dropout(),
            nn.Linear(1024,1)
        )
    def forward(self, graph_data: PyGData):
        xc, xd = graph_data.cell, graph_data.drug
        xc2 = self.cell_conv(xc)
        xd2 = self.drug_conv(xd)
        x_all = torch.cat([xc2,xd2])
        out = self.fc(x_all)
        return out


    
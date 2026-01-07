# gnn_model.py
import torch
from torch.nn import Linear, Dropout, Sequential, ReLU
from torch_geometric.nn import NNConv, TransformerConv
from torch import nn
from torch_geometric.nn.models.tgn import TGNMemory, IdentityMessage, LastAggregator
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops, add_remaining_self_loops, coalesce

class SpatioTemporalGCN(torch.nn.Module):
    def __init__(self, in_channels, edge_attr_dim, hidden_channels=64,
                 dropout_in=0.05, dropout_hidden=0.1):
        super().__init__()
        self.dropout_in = Dropout(p=dropout_in)
        self.dropout_hidden = Dropout(p=dropout_hidden)

        # Define neural networks for edge-conditioned convolutions
        nn1 = Sequential(
            Linear(edge_attr_dim, 64), ReLU(),
            Linear(64, in_channels * hidden_channels)
        )
        nn2 = Sequential(
            Linear(edge_attr_dim, 64), ReLU(),
            Linear(64, hidden_channels * hidden_channels)
        )

        # Define the graph convolutional layers
        self.conv1 = NNConv(in_channels, hidden_channels, nn1, aggr='mean')
        self.conv2 = NNConv(hidden_channels, hidden_channels, nn2, aggr='mean')

        # Define the output layers
        self.lin1 = Linear(hidden_channels, 8)
        self.lin2 = Linear(8, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.zeros_(self.lin1.bias)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        torch.nn.init.zeros_(self.lin2.bias)

    def forward(self, x, edge_index, edge_attr):
        x = self.dropout_in(x)
        x = torch.relu_(self.conv1(x, edge_index, edge_attr))
        x = self.dropout_hidden(x)
        x = torch.relu_(self.conv2(x, edge_index, edge_attr))
        x = self.dropout_hidden(x)
        x = torch.relu_(self.lin1(x))
        return self.lin2(x).squeeze(-1)
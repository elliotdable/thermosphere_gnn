# gnn_model.py
import torch
from torch.nn import Linear, Dropout
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_remaining_self_loops, coalesce

class SpatioTemporalGAT(torch.nn.Module):
    def __init__(self, in_channels, edge_attr_dim, hidden_channels=64,
                 heads=4, dropout_in=0.05, dropout_hidden=0.1, learn_self_loops=True):
        super().__init__()

        self.dropout_in = Dropout(p=dropout_in)
        self.dropout_hidden = Dropout(p=dropout_hidden)

        self.edge_attr_dim = edge_attr_dim

        # 1D vector used for ALL self-loop edges (safe for PyG fill_value)
        if learn_self_loops:
            self.self_loop_attr = torch.nn.Parameter(torch.zeros(edge_attr_dim))
        else:
            self.register_buffer("self_loop_attr", torch.zeros(edge_attr_dim))

        self.conv1 = GATConv(
            in_channels,
            hidden_channels // heads,
            heads=heads,
            edge_dim=edge_attr_dim,
            dropout=dropout_hidden,
            add_self_loops=False
        )

        self.conv2 = GATConv(
            hidden_channels,
            hidden_channels // heads,
            heads=heads,
            edge_dim=edge_attr_dim,
            dropout=dropout_hidden,
            add_self_loops=False
        )

        self.lin1 = Linear(hidden_channels, 8)
        self.lin2 = Linear(8, 1)

    def _add_self_loops_safe(self, x, edge_index, edge_attr):
        num_nodes = x.size(0)

        # Make sure these are CUDA-kernel friendly
        edge_index = edge_index.contiguous()
        if edge_attr is not None:
            edge_attr = edge_attr.contiguous()

        # Add loops only where missing; if loops already exist, their edge_attr is preserved.
        # New loop edge_attr rows are filled with self.self_loop_attr (1D, broadcasted).
        edge_index, edge_attr = add_remaining_self_loops(
            edge_index,
            edge_attr=edge_attr,
            fill_value=self.self_loop_attr,   # IMPORTANT: 1D (edge_attr_dim,)
            num_nodes=num_nodes
        )

        # Coalesce to ensure sorted + dedup (helps some CUDA paths)
        if edge_attr is not None:
            edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes=num_nodes)
        else:
            edge_index, _ = coalesce(edge_index, None, num_nodes=num_nodes)

        return edge_index, edge_attr

    def _assert_batch_ok(self, x, edge_index, edge_attr, where=""):
        N = x.size(0)
        E = edge_index.size(1)

        assert edge_index.dtype == torch.long, f"{where}: edge_index dtype {edge_index.dtype}"
        assert edge_index.numel() > 0, f"{where}: empty edge_index"
        assert edge_index.min().item() >= 0, f"{where}: edge_index has negatives: {edge_index.min().item()}"
        assert edge_index.max().item() < N, f"{where}: edge_index max {edge_index.max().item()} >= N {N}"

        if edge_attr is not None:
            assert edge_attr.size(0) == E, f"{where}: edge_attr rows {edge_attr.size(0)} != E {E}"
            assert edge_attr.size(1) == self.edge_attr_dim, f"{where}: edge_attr dim {edge_attr.size(1)} != {self.edge_attr_dim}"
            assert torch.isfinite(edge_attr).all(), f"{where}: edge_attr has NaN/Inf"

        assert torch.isfinite(x).all(), f"{where}: x has NaN/Inf"

    def forward(self, x, edge_index, edge_attr):
        edge_index, edge_attr = self._add_self_loops_safe(x, edge_index, edge_attr)
        self._assert_batch_ok(x, edge_index, edge_attr, where="pre-conv1")

        x = self.dropout_in(x)
        x = torch.relu(self.conv1(x, edge_index, edge_attr))
        x = self.dropout_hidden(x)

        self._assert_batch_ok(x, edge_index, edge_attr, where="pre-conv2")
        x = torch.relu(self.conv2(x, edge_index, edge_attr))
        x = self.dropout_hidden(x)

        x = torch.relu(self.lin1(x))
        return self.lin2(x).squeeze(-1)


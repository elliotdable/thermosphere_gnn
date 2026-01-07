# gnn_model.py
import torch
from torch.nn import Linear, Dropout
from torch_geometric.nn import TransformerConv
from torch import nn
from torch_geometric.nn.models.tgn import TGNMemory, IdentityMessage, LastAggregator

class FixedTimeEncoder(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.lin = nn.Linear(1, out_channels)

    def forward(self, t):
        t = t.view(-1, 1).float()
        return torch.cos(self.lin(t))

class SpatioTemporalTGN(nn.Module):
    def __init__(self, in_channels, edge_attr_dim, hidden_channels=64,
                 dropout_in=0.05, dropout_hidden=0.1):
        super().__init__()

        self.hidden_channels = hidden_channels

        # Project raw node features -> hidden dim
        self.x_encoder = Linear(in_channels, hidden_channels)

        self.memory = None
        self.time_encoder = FixedTimeEncoder(hidden_channels)

        self.conv = TransformerConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            heads=1,
            concat=False,
            dropout=0.1,
            edge_dim=edge_attr_dim
        )

        self.lin1 = Linear(hidden_channels, 8)
        self.lin2 = Linear(8, 1)
        self.dropout_in = Dropout(dropout_in)
        self.dropout_hidden = Dropout(dropout_hidden)
        self.hidden_channels = hidden_channels


    def reset_memory(self):
        self.memory.reset_state()

    def set_num_nodes(self, num_nodes: int):
        """
        Now that we know the real number of nodes, build the TGN memory correctly.
        """
        self.memory = TGNMemory(
            num_nodes=num_nodes,
            raw_msg_dim=self.hidden_channels,
            memory_dim=self.hidden_channels,
            time_dim=self.hidden_channels,
            message_module=IdentityMessage(
                self.hidden_channels,
                self.hidden_channels,
                self.hidden_channels
            ),
            aggregator_module=LastAggregator()
        )
        self.memory.reset_state()
        
    def forward(self, src, dst, t, x, edge_attr):
        # Encode node features to hidden space
        x = self.dropout_in(self.x_encoder(x))

        # Retrieve memory for involved nodes
        z_src = self.memory.memory[src]
        z_dst = self.memory.memory[dst]

        # Time encoder MUST run in full precision (fp32)
        t = t.float()

        with torch.cuda.amp.autocast(enabled=False):
            # Ensure timestamps are float32
            t = t.to(dtype=torch.float32)

            # Safest – run TimeEncoder on CPU (slow but very robust)
            # Comment this back out once things are stable.
            t_cpu = t.detach().cpu().view(-1, 1)
            dt_enc_cpu = self.time_encoder(t_cpu)  # this is a plain nn.Sequential+Linear
            dt_enc = dt_enc_cpu.to(t.device)
            
        messages = z_src + dt_enc
        raw_msg = messages

        # Update memory
        self.memory.update_state(src, dst, raw_msg, t)
        self.memory.update_state(src, dst, raw_msg, t)

        # Convolution uses GLOBAL graph
        edge_index = torch.stack([src, dst], dim=0)

        # (we’ll fix the x dimension next)
        h_local = self.conv(x, edge_index, edge_attr)

        h = h_local[dst]

        h = torch.relu(self.lin1(h))
        out = self.lin2(h).squeeze(-1)
        return out

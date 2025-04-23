import torch
from torch_geometric.nn import GATv2Conv, global_max_pool
import torch.nn.functional as F
from torch.nn import LayerNorm
import torch.nn as nn


class GraphTransformer(torch.nn.Module):
    def __init__(self, num_classes, in_channels, out_channels, heads=1, dropout=0.0, edge_dim=384):
        super(GraphTransformer, self).__init__()

        # Layer 1: Propagate from C to B
        self.transformer1 = GATv2Conv(
            in_channels, out_channels, heads=heads, dropout=dropout, edge_dim=edge_dim, flow="target_to_source",
            add_self_loops=True
        )

        # Layer 2: Propagate from B to A
        self.transformer2 = GATv2Conv(
            out_channels * heads, out_channels, heads=1, dropout=dropout, edge_dim=edge_dim, flow="target_to_source",
            add_self_loops=True
        )

        # Layer normalization
        self.layer_norm = LayerNorm(out_channels)
        self.num_classes = num_classes
        self.classifier = nn.Linear(out_channels, self.num_classes)

        self.att = []
        

    def forward(self, data):
        # Input: Node features (x), edge index, and edge attributes
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Layer 1: First attention layer
        x1, attention_weights1 = self.transformer1(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        x1 = F.leaky_relu(x1)

        # Layer 2: Second attention layer
        x2, attention_weights2 = self.transformer2(x1, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        x2 = F.leaky_relu(x2)

        # Pooling to get graph-level representation
        x2 = global_max_pool(x2, data.batch)

        # Layer normalization
        x2 = self.layer_norm(x2)

        if self.training:
            cls_score = self.classifier(x2)
            return cls_score, x2  # global feature for triplet loss
        
        self.att = [attention_weights1, attention_weights2]

        return x2, attention_weights1, attention_weights2  # Return both graph-level representation and attention
        # weights

    def __str__(self):
        description = (
            "GraphTransformer Model\n"
            "=======================\n"
            f"Input channels: {self.transformer1.in_channels}\n"
            f"Output channels: {self.transformer1.out_channels}\n"
            f"Attention heads: {self.transformer1.heads}\n"
            f"Edge dimension: {self.transformer1.edge_dim}\n"
            f"Dropout: {0.0}\n"
            f"add_self_loops: True\n"
            "Layers:\n"
            "  1. GATv2Conv (C -> B): First attention mechanism\n"
            "  2. GATv2Conv (B -> A): Second attention mechanism\n"
            "  3. Global Max Pooling: Aggregates node features to graph-level representation\n"
            f"  4. Layer Normalization: Applies to graph-level features with shape {self.layer_norm.normalized_shape}\n"
        )
        return description

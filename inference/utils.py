# encoding: utf-8
"""
@author:  raufschlaeger
"""

import os
import sys

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import numpy as np

import networkx as nx
import torch
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from sentence_transformers import SentenceTransformer

from config import cfg
from data.src.processing.scene_graph import SceneGraph

from data.src.utils.graph_plotting import plot_graph
from modeling.graph_transformer import GraphTransformer

from modeling import build_model

import torch
import random
import numpy as np

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)


from gatt import get_gatt

from gatt import get_gatt, get_avgatt
from vis_utils import (
    draw_local_comp_graph_with_attribution_scores_BAShapes,
)


def visualize_gatt(model, graph, node_strings, base_name=None):

    scene_graph = SceneGraph(graph, SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'))

    # Convert the graph to tensor data
    sample = scene_graph.nx_to_tg_data().to(device="cuda")

    # Forward pass to get the attention weights
    with torch.no_grad():
        graph_representation, attention_weights1, attention_weights = model(sample)

    # revert
    sample.edge_index = sample.edge_index[[1, 0], :]

    target_node_l2 = 0
    gatt_val_l2, edge_index_l2 = get_gatt(
        target_node=target_node_l2, model=model, data=sample, sparse=True
    )
    print(f"GAtt values for GAT_L2 (showing all): {gatt_val_l2[:]}")

    edge_index_l2 = torch.Tensor(edge_index_l2).long().t()
    att_matrix_l2 = torch.zeros((sample.num_nodes, sample.num_nodes)).to("cuda")
    att_matrix_l2[edge_index_l2[1], edge_index_l2[0]] = torch.tensor(gatt_val_l2).to("cuda")


    draw_local_comp_graph_with_attribution_scores_BAShapes(
        data=sample,
        hops=2,
        target_idx=target_node_l2,
        att_matrix=att_matrix_l2,
        node_strings=node_strings,
        base_name=base_name
    )


def extract_node_strings(graph):
    """
    Extract node strings in the same order they are added in _build_graph.
    """
    import json
    graph_data = json.loads(graph)
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    node_strings = []

    for node in nodes:
        node_strings.append(node['id'])  # Add the node ID
        attributes = node.get('attributes', [])
        if isinstance(attributes, list):
            for attr in attributes:
                if isinstance(attr, dict):
                    attr_str = ' '.join(str(value) for value in attr.values())
                    node_strings.append(attr_str)
                else:
                    node_strings.append(attr)
        elif isinstance(attributes, dict):
            attr_str = ' '.join(str(value) for value in attributes.values())
            node_strings.append(attr_str)

    return node_strings


def visualize_attention_weights(model, graph, node_strings, base_name=None):
    """
    Visualize raw attention weights from the model's output.
    """
    scene_graph = SceneGraph(graph, SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'))
    sample = scene_graph.nx_to_tg_data().to(device="cuda")

    with torch.no_grad():
        graph_representation, attention_weights1, attention_weights = model(sample)

    # plot atts in layer 2:
    # Extract edges (pairs of nodes) and attention weights
    edges = attention_weights[0].cpu().numpy()  # Shape: (num_edges, 2)
    weights = attention_weights[1].cpu().numpy()  # Shape: (num_edges, 1)

    edges = edges.T  # Shape (num_edges, 2)
    weights = weights.flatten()  # Shape (num_edges,)

    # Create a graph
    G = nx.DiGraph()
    for (src, dst), weight in zip(edges, weights):
        G.add_edge(dst, src, weight=weight)  # Reverse edges for dataflow

    # Normalize the weights and create a colormap
    norm = Normalize(vmin=min(weights), vmax=max(weights))
    cmap = plt.cm.viridis  # Choose a colormap
    sm = ScalarMappable(cmap=cmap, norm=norm)

    # Define the figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))  # Compact figure

    pos = nx.circular_layout(G)
    # Prepare node labels with both index and node string
    node_labels = {}
    for idx in G.nodes():
        if node_strings and idx < len(node_strings):
            node_labels[idx] = f"{idx}: {node_strings[idx]}"
        else:
            node_labels[idx] = str(idx)

    # Map edge weights to colors
    edge_colors = [sm.to_rgba(G[u][v]['weight']) for u, v in G.edges()]

    # Draw nodes and edges
    nx.draw(
        G,
        pos,
        ax=ax,
        with_labels=True,
        labels=node_labels,
        node_color="lightblue",
        node_size=250,
        font_size=12,
        edge_color=edge_colors,
        width=1.5,
        connectionstyle="arc3,rad=0.15"
    )

    # Add edge labels (attention weights)
    edge_labels = {edge: f"{G[edge[0]][edge[1]]['weight']:.2f}" for edge in G.edges()}
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_color="darkgreen",
        font_size=9,
        label_pos=0.3,
        ax=ax
    )

    # Add a colorbar for the weights
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", label="Attention Weight")

    # Set plot title and adjust layout
    plt.title("Attention Weights Visualization")
    plt.tight_layout(pad=4.0)  # Increase padding for more border space
    plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)  # Increase border size

    # Save the plot to a file
    if base_name is None:
        plot_filename = f"inference/att_weights_example.png"
    else:
        plot_filename = os.path.basename("./inference/gatt_graphs/" + base_name)

    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')


# Example Usage
if __name__ == "__main__":

    folder_path = 'inference'

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")


    # graph = ' {\n  "nodes": [\n    {\n      "id": "person",\n      "attributes": ["black hair", "black shirt", "black pants", "black shoes", "holding phone", "holding drink"]\n    },\n    {\n      "id": "phone",\n      "attributes": ["in hand"]\n    },\n    {\n      "id": "drink",\n      "attributes": ["in hand"]\n    },\n    {\n      "id": "ground",\n      "attributes": ["visible"]\n    },\n    {\n      "id": "background",\n      "attributes": ["pink sign", "yellow text"]\n    }\n  ],\n  "edges": [\n    {\n      "source": "person",\n      "target": "phone"\n    },\n    {\n      "source": "person",\n      "target": "drink"\n    },\n    {\n      "source": "ground",\n      "target": "person"\n    },\n    {\n      "source": "background",\n      "target": "person"\n    }\n  ]\n}'
    # graph = ' {\n  "nodes": [\n    {\n      "id": "person",\n      "attributes": ["black and white checkered shirt", "pink backpack", "black leggings", "white shoes"]\n    },\n    {\n      "id": "umbrella",\n      "attributes": ["purple"]\n    },\n    {\n      "id": "headwear",\n      "attributes": ["black", "white"]\n    }\n  ],\n  "edges": [\n    {\n      "source": "person",\n      "target": "umbrella"\n    },\n    {\n      "source": "person",\n      "target": "headwear"\n    },\n    {\n      "source": "umbrella",\n      "target": "headwear"\n    }\n  ]\n}'
    # graph = ' {\n  "nodes": [\n    {\n      "id": "person",\n      "attributes": ["long black hair", "white backpack", "white shoes", "holding umbrella"]\n    },\n    {\n      "id": "umbrella",\n      "attributes": ["pink"]\n    },\n    {\n      "id": "bike",\n      "attributes": ["black"]\n    },\n    {\n      "id": "road",\n      "attributes": ["gray"]\n    },\n    {\n      "id": "bushes",\n      "attributes": ["green"]\n    },\n    {\n      "id": "tree",\n      "attributes": ["brown"]\n    },\n    {\n      "id": "grass",\n      "attributes": ["green"]\n    }\n  ],\n  "edges": [\n    {\n      "source": "person",\n      "target": "umbrella"\n    },\n    {\n      "source": "person",\n      "target": "bike"\n    },\n    {\n      "source": "bike",\n      "target": "road"\n    },\n    {\n      "source": "bushes",\n      "target": "grass"\n    },\n    {\n      "source": "tree",\n      "target": "bushes"\n    }\n  ]\n}'
    # graph = ' {\n  "nodes": [\n    {\n      "id": "person",\n      "attributes": ["white t-shirt", "black shorts", "black shoes", "short black hair", "male", "running"]\n    },\n    {\n      "id": "gray wall",\n      "attributes": ["background"]\n    },\n    {\n      "id": "black rectangle",\n      "attributes": ["on ground"]\n    }\n  ],\n  "edges": [\n    {\n      "source": "person",\n      "target": "gray wall",\n      "relation": "in front of"\n    },\n    {\n      "source": "person",\n      "target": "black rectangle",\n      "relation": "running towards"\n    }\n  ]\n}'
    # graph = ' {\n  "nodes": [\n    {\n      "id": "person",\n      "attributes": ["white shirt", "black pants", "black hat", "black strap over shoulder", "walking towards camera"]\n    },\n    {\n      "id": "concrete",\n      "attributes": ["gray surface"]\n    },\n    {\n      "id": "green object",\n      "attributes": ["behind person", "blue square on it"]\n    }\n  ],\n  "edges": [\n    {\n      "source": "person",\n      "target": "concrete",\n      "relation": "standing on"\n    },\n    {\n      "source": "green object",\n      "target": "person",\n      "relation": "behind"\n    }\n  ]\n}'

    # # market - train - 0729_c6s2_057343_03.jpg,729,6,5,3
    # graph = ' {\n  "nodes": [\n    {\n      "id": "person",\n      "attributes": ["orange shirt", "black backpack", "blue shorts", "black shoes", "black hair"]\n    },\n    {\n      "id": "door",\n      "attributes": ["black", "wood"]\n    },\n    {\n      "id": "ground",\n      "attributes": ["gray"]\n    }\n  ],\n  "edges": [\n    {\n      "source": "person",\n      "target": "door",\n      "relation": "standing in front of"\n    },\n    {\n      "source": "person",\n      "target": "ground",\n      "relation": "standing on"\n    }\n  ]\n}'

    # market - train - 0629_c6s2_029568_02.jpg,629,6,2,2," {
    graph = ' {\n  "nodes": [\n    {\n      "id": "person",\n      "attributes": ["long brown hair", "glasses", "blue t-shirt", "white square on shirt", "green shorts", "black shoes"]\n    },\n    {\n      "id": "phone",\n      "attributes": ["held in right hand"]\n    },\n    {\n      "id": "yellow object",\n      "attributes": ["held in left hand"]\n    }\n  ],\n  "edges": [\n    {\n      "source": "person",\n      "target": "phone",\n      "relation": "holding"\n    },\n    {\n      "source": "person",\n      "target": "yellow object",\n      "relation": "holding"\n    }\n  ]\n}'


    config_file = './configs/market1501/gat_softmax_triplet_with_center.yml'
    cfg.merge_from_file(config_file)
    
    model = build_model(cfg, 751).to(device="cuda")
    model.training = False
    model.eval()

    
    config_file = './configs/market1501/gat_dinov2_vitb14_softmax_triplet_with_center.yml'
    cfg.merge_from_file(config_file)
    teacher_model = build_model(cfg, 751).to(device="cuda")

    # Print all keys in teacher_model's state_dict
    print("Keys in teacher_model.state_dict():")
    for k in teacher_model.state_dict().keys():
        print(k)

    # Remove "gat." prefix from all keys in teacher_model's state_dict
    teacher_state = teacher_model.state_dict()
    teacher_state_no_gat = {}
    for k, v in teacher_state.items():
        if k.startswith("gat."):
            teacher_state_no_gat[k[4:]] = v
        else:
            teacher_state_no_gat[k] = v

    # Load all weights from teacher_model into model, allowing for missing/unexpected keys
    missing = model.load_state_dict(teacher_state_no_gat, strict=False)
    # print("Missing keys:", missing.missing_keys)
    # print("Unexpected keys:", missing.unexpected_keys)
    # Print only the keys that were successfully loaded (i.e., present in both model and teacher_state_no_gat)
    loaded_keys = [k for k in teacher_state_no_gat.keys() if k in model.state_dict().keys()]
    print("Successfully loaded keys:", loaded_keys)

    # Extract node strings
    node_strings = extract_node_strings(graph)
    print("Extracted Node Strings:", node_strings)

    # Perform inference
    # visualize_gatt(model, graph, node_strings)
    visualize_attention_weights(model, graph, node_strings)

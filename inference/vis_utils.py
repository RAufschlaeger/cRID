# adapted from: https://github.com/jordan7186/GAtt

import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, add_self_loops
import os


# Let's turn the process of drawing the local graph with the ground truth path into a function
def draw_local_comp_graph_with_ground_truth_path_Infection(
    data: Data, hops: int, target_idx: int, ground_truth: bool = True
) -> None:
    # First assert that the target index does have a unique ground truth path
    assert (
        target_idx in data.unique_solution_nodes
    ), "Target index does not have a unique ground truth path"
    # Get the local k hop subgraph
    subgraph_nodes, _, _, inv = k_hop_subgraph(
        node_idx=target_idx,
        num_hops=hops,
        edge_index=data.edge_index,
        relabel_nodes=True,
    )
    # Convert nodes and edges to lists
    subgraph_nodes = subgraph_nodes.tolist()
    subgraph_edges = data.edge_index[:, inv].tolist()
    # Transform subgraph_edges to a list of tuples
    subgraph_edges_tup = [
        (subgraph_edges[0][i], subgraph_edges[1][i])
        for i in range(len(subgraph_edges[0]))
    ]
    # Create a directed graph
    G = nx.DiGraph()
    G.add_edges_from(subgraph_edges_tup)
    # Remove self-loops from the graph
    G.remove_edges_from(nx.selfloop_edges(G))
    if ground_truth:
        # Get index of target node in data.unique_solution_nodes
        target_idx_in_unique_solution_nodes = data.unique_solution_nodes.index(
            target_idx
        )
        # Get the ground truth path for target node
        ground_truth_path = data.unique_solution_explanations[
            target_idx_in_unique_solution_nodes
        ]
        # Convert the ground truth path to a list of tuples
        ground_truth_path_tup = [
            (ground_truth_path[i], ground_truth_path[i + 1])
            for i in range(len(ground_truth_path) - 1)
        ]
    # Draw the graph with subgraph_nodes as node labels
    # Highlight the path from ground_truth_path_tup with red edges
    # Also highlight the target node with a different color
    plt.figure(figsize=(6, 6), dpi=100)
    pos = nx.spring_layout(G, seed=0)
    node_size = 800
    nx.draw(
        G,
        pos=pos,
        node_color="#D8D4F2",
        node_size=node_size,
        width=2,
        edgecolors="black",
        linewidths=2,
        edge_color="black",
        arrowstyle="-|>",
        labels={node: node for node in subgraph_nodes},  # Add node labels
        with_labels=True,
        font_weight="normal",
    )
    if ground_truth:
        nx.draw_networkx_edges(
            G,
            pos=pos,
            edgelist=ground_truth_path_tup,
            edge_color="#5152d0",
            width=6,
            arrows=True,
            arrowsize=25,
            node_size=node_size,
        )
    nx.draw_networkx_nodes(
        G,
        pos=pos,
        nodelist=[target_idx],
        node_color="#FF8C00",
        node_size=node_size,
        linewidths=2,
        edgecolors="black",
    )

    plt.show()
    output_path = os.path.join(os.getcwd(), 'inference/local_comp_graph.png')
    print(output_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=1.0)  # Add padding to avoid cutting text


def draw_local_comp_graph_with_attribution_score_Infection(
    data: Data, hops: int, target_idx: int, att_matrix: torch.Tensor
) -> None:
    # First assert that the target index does have a unique ground truth path
    assert (
        target_idx in data.unique_solution_nodes
    ), "Target index does not have a unique ground truth path"
    # Get the local k hop subgraph
    subgraph_nodes, _, _, inv = k_hop_subgraph(
        node_idx=target_idx,
        num_hops=hops,
        edge_index=data.edge_index,
        relabel_nodes=True,
    )
    # Convert nodes and edges to lists
    subgraph_nodes = subgraph_nodes.tolist()
    subgraph_edges = data.edge_index[:, inv].tolist()
    # Transform subgraph_edges to a list of tuples
    subgraph_edges_tup = [
        (subgraph_edges[0][i], subgraph_edges[1][i])
        for i in range(len(subgraph_edges[0]))
    ]
    # Add self-loops to the graph
    subgraph_edges_tup = subgraph_edges_tup + [(node, node) for node in subgraph_nodes]

    # Create a directed graph
    G = nx.DiGraph()
    G.add_edges_from(subgraph_edges_tup)

    # Draw the graph with subgraph_nodes as node labels
    # Also highlight the target node with a different color
    plt.figure(figsize=(6, 6), dpi=100)
    pos = nx.spring_layout(G, seed=0)
    node_size = 800
    nx.draw(
        G,
        pos=pos,
        node_color="#D8D4F2",
        node_size=node_size,
        width=1,
        edgecolors="black",
        linewidths=2,
        edge_color="white",
        arrowstyle="-|>",
        labels={node: node for node in subgraph_nodes},  # Add node labels
        with_labels=True,
        font_weight="normal",
    )

    # Get the edge attribution scores from att_matrix
    # Edge (i,j) is weighted by att_matrix[j,i]
    # Normalize att_matrix to [0,0.8] for better visualization
    # att_matrix = att_matrix / att_matrix.max()

    edge_color = [att_matrix[edge[1], edge[0]].item() for edge in subgraph_edges_tup]

    edge_labels = {
        (edge[0], edge[1]): f"{att_matrix[edge[1], edge[0]].item():.2f}"
        for edge in subgraph_edges_tup
    }
    nx.draw_networkx_edges(
        G,
        pos=pos,
        edgelist=subgraph_edges_tup,
        edge_color=edge_color,
        edge_vmin=0,
        edge_vmax=1,
        width=2,
        edge_cmap=plt.cm.OrRd,
        min_target_margin=12,
        min_source_margin=12,
    )
    nx.draw_networkx_edge_labels(
        G,
        pos=pos,
        edge_labels=edge_labels,  # Add edge labels for attribution scores
        font_size=15,
        font_color="black",  # Set text color for edge labels
    )
    nx.draw_networkx_nodes(
        G,
        pos=pos,
        nodelist=[target_idx],
        node_color="#FF8C00",
        node_size=node_size,
        edgecolors="black",
        linewidths=2,
        label={node: node for node in subgraph_nodes},
    )
    plt.show()


def draw_local_comp_graph_with_ground_truth_house_BAShapes(
    data: Data, target_idx: int, num_hops: int
):
    """
    Draw the local computation graph of a target node with the ground truth edges
    """
    subgraph_nodes, _, _, inv = k_hop_subgraph(
        node_idx=target_idx,
        num_hops=num_hops,
        edge_index=data.edge_index,
        relabel_nodes=True,
    )

    # Make a new mask that is True only when inv AND data.edge_mask is true
    ground_truth_mask = data.edge_mask.bool() & inv
    # Nodes with index larger or equal to 300 are all ground truth nodes
    ground_truth_nodes_mask = subgraph_nodes >= 300
    ground_truth_nodes = subgraph_nodes[ground_truth_nodes_mask].tolist()
    # Convert nodes and edges to lists
    subgraph_nodes = subgraph_nodes.tolist()
    subgraph_edges = data.edge_index[:, inv].tolist()
    ground_truth_edges = data.edge_index[:, ground_truth_mask].tolist()
    # Transform subgraph_edges to a list of tuples
    subgraph_edges_tup = [
        (subgraph_edges[0][i], subgraph_edges[1][i])
        for i in range(len(subgraph_edges[0]))
    ]
    # Transform ground_truth_edges to a list of tuples
    ground_truth_edges_tup = [
        (ground_truth_edges[0][i], ground_truth_edges[1][i])
        for i in range(len(ground_truth_edges[0]))
    ]

    # Create a directed graph
    G = nx.DiGraph()
    G.add_edges_from(subgraph_edges_tup)
    # Remove self-loops from the graph
    G.remove_edges_from(nx.selfloop_edges(G))
    pos = nx.spring_layout(G, seed=0)

    plt.figure(figsize=(6, 6), dpi=100)
    node_size = 800
    nx.draw(
        G,
        pos=pos,
        node_color="#D8D4F2",
        node_size=node_size,
        font_size=30,
        width=1.5,
        edgecolors="black",
        linewidths=2,
        edge_color="black",
        arrowstyle="-|>",
        labels={node: node for node in subgraph_nodes},  # Add node labels
        with_labels=True,
    )

    nx.draw_networkx_edges(
        G,
        pos=pos,
        edgelist=ground_truth_edges_tup,
        edge_color="#5152d0",
        width=5,
        arrows=True,
        arrowsize=25,
        node_size=node_size,
    )
    nx.draw_networkx_nodes(
        G,
        pos=pos,
        nodelist=ground_truth_nodes,
        node_color="#5152d0",
        node_size=node_size,
        linewidths=2,
        edgecolors="black",
    )
    nx.draw_networkx_nodes(
        G,
        pos=pos,
        nodelist=[target_idx],
        node_color="#FF8C00",
        node_size=node_size,
        linewidths=2,
        edgecolors="black",
    )
    plt.show()


def draw_local_comp_graph_with_attribution_scores_BAShapes(
    data: Data, hops: int, target_idx: int, att_matrix: torch.Tensor, node_strings=None, base_name=None
) -> None:
    edge_index = add_self_loops(data.edge_index)[0]
    # Get the local k hop subgraph
    subgraph_nodes, _, _, inv = k_hop_subgraph(
        node_idx=target_idx,
        num_hops=hops,
        edge_index=edge_index,
        relabel_nodes=True,
    )
    # Nodes with index larger or equal to 300 are all ground truth nodes
    ground_truth_nodes_mask = subgraph_nodes >= 300
    ground_truth_nodes = subgraph_nodes[ground_truth_nodes_mask].tolist()
    # Convert nodes and edges to lists
    subgraph_nodes = subgraph_nodes.tolist()
    subgraph_edges = edge_index[:, inv].tolist()
    # Transform subgraph_edges to a list of tuples
    subgraph_edges_tup = [
        (subgraph_edges[0][i], subgraph_edges[1][i])
        for i in range(len(subgraph_edges[0]))
    ]
    # Create a directed graph
    G = nx.DiGraph()
    G.add_edges_from(subgraph_edges_tup)

    # Prepare node labels with both index and node string if provided
    if node_strings:
        node_labels = {}
        for idx in subgraph_nodes:
            if idx < len(node_strings):
                label = f"{idx}: {node_strings[idx]}"
                # Split label into two lines if longer than 100 chars
                if len(label) > 100:
                    # Try to split at the first space after 12 chars, else just split at 15
                    split_pos = label.find(' ', 12)
                    if split_pos == -1 or split_pos > 30:
                        split_pos = 15
                    label = label[:split_pos] + '\n' + label[split_pos:].lstrip()
                node_labels[idx] = label
            else:
                node_labels[idx] = str(idx)
    else:
        node_labels = {node: node for node in subgraph_nodes}

    # Draw the graph with subgraph_nodes as node labels
    # Also highlight the target node with a different color
    plt.figure(figsize=(8, 8), dpi=300)
    # Increase k and iterations to spread nodes further apart and reduce label overlap
    pos = nx.spring_layout(G, seed=0, k=1.5, iterations=200)
    node_size = 1000
    arc_rad = 0.05


    nx.draw_networkx_nodes(
        G,
        pos=pos,
        nodelist=ground_truth_nodes,
        node_color="#5152d0",
        node_size=node_size,
        linewidths=2,
        edgecolors="gray",
    )
    nx.draw_networkx_nodes(
        G,
        pos=pos,
        nodelist=[target_idx],
        node_color="#FFD9B3",  # lighter orange
        node_size=node_size,
        edgecolors="gray",
        linewidths=2,
        label={node: node for node in subgraph_nodes},
    )

    nx.draw(
        G,
        pos=pos,
        node_color="#F3F3FD",  # lighter color
        node_size=node_size,
        width=3,
        edgecolors="gray",
        linewidths=2,
        edge_color="white",
        arrowstyle="-|>",
        connectionstyle=f"arc3, rad = {arc_rad}",
        labels=None,  # Draw labels separately to control bbox
        with_labels=False,
        font_weight="normal",
        font_size=15
    )

    # Draw node labels with a white bounding box to improve readability and reduce overlap
    nx.draw_networkx_labels(
        G,
        pos=pos,
        labels=node_labels,
        font_size=15,
        font_weight="normal",
        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2')
    )

    edge_color = [att_matrix[edge[1], edge[0]].item() for edge in subgraph_edges_tup]

    edge_labels = {
        (edge[0], edge[1]): f"{att_matrix[edge[1], edge[0]].item():.3f}"
        for edge in subgraph_edges_tup
    }
    nx.draw_networkx_edges(
        G,
        pos=pos,
        edgelist=subgraph_edges_tup,
        edge_color=edge_color,
        edge_vmin=0,
        edge_vmax=1,
        width=2,
        edge_cmap=plt.cm.OrRd,
        connectionstyle=f"arc3, rad = {arc_rad}",
        min_target_margin=15,
        min_source_margin=15,
    )

    # Save the plot to a file
    if base_name is None:
        plot_filename = f"inference/gatt_graph_example.png"
    else:
        plot_filename = "./inference/gatt_graphs/" + base_name

    plt.savefig(plot_filename, dpi=400, pad_inches=1.0)  # Add padding to avoid cutting text

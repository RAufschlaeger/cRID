# adapted from: https://github.com/jordan7186/GAtt

# Tools for analyzing attention weights
import torch
from torch_geometric.utils import remove_self_loops, add_self_loops, get_num_hops
from torch_geometric.data import Data
from typing import List, Tuple, Dict
import networkx as nx
from torch_geometric.utils import (
    remove_self_loops,
    add_self_loops,
    get_num_hops,
    k_hop_subgraph,
)


def return_edges_in_k_hop(
    data: Data,
    target_idx: int,
    hop: int,
    self_loops: bool = False,
    return_as_tensor: bool = False,
) -> List[Tuple[int, int]]:
    r"""Returns all edges in :obj:`data` that are connected to :obj:`target_idx`
    and lie within a :obj:`hop` distance.

    Args:
        data (Data): The graph data object.
        target_idx (int): The central node.
        hop (int): The number of hops.
        add_self_loops (bool, optional): If set to :obj:`True`, will add self-loops
            in the returned edge indices. (default: :obj:`False`)
    """
    assert hop > 0
    if self_loops:
        edge_index = add_self_loops(remove_self_loops(data.edge_index)[0])[0]
    else:
        edge_index = remove_self_loops(data.edge_index)[0]

    _, _, _, inv = k_hop_subgraph(
        node_idx=target_idx,
        num_hops=hop,
        edge_index=edge_index,
        relabel_nodes=True,
    )

    if return_as_tensor:
        return edge_index[:, inv]
    else:
        return edge_index[:, inv].t().tolist()


@torch.no_grad()
def generate_att_dict(model, data, sparse: bool = True) -> Dict:
    """
    Generates a dictionary of attention matrices from a model.
    Attention heads are averaged across all heads.
    Args:
        model: The GAT model.
        data: The data object.
        sparse: Whether to return the attention matrices in sparse format.

    Returns:
        A dictionary of attention matrices
    """
    _, att1, att2 = model(data)
    num_nodes = data.num_nodes
    att = [att1, att2]
    att_matrix_dict = {}
    device = "cuda"
    for idx, att_info in enumerate(att):
        if sparse:
            att_matrix_dict[idx] = torch.sparse_coo_tensor(
                att_info[0],
                att_info[1].mean(dim=1).squeeze(),
                size=(num_nodes, num_nodes),
                device=device,
            ).t()
        else:
            att_matrix_dict[idx] = torch.zeros((num_nodes, num_nodes)).to(device)
            att_matrix_dict[idx][att_info[0][1], att_info[0][0]] = (
                att_info[1].mean(dim=1).squeeze()
            )
    return att_matrix_dict


def prep_for_gatt(model, data, num_hops: int, sparse: bool = True) -> Tuple[Dict, Dict]:
    """
    Calculates attention matrices that will be used for actual computatoin of GAtt.
    Main function is to calculate the correction matrix (C matrix in the original paper).
    Args:
        model: The GAT model.
        data: The data object.
        num_hops: The number of layers of the GAT model.
    Returns:
        A tuple of attention matrix dictionary and correction matrix dictionary.
    """
    att_matrix_dict = generate_att_dict(model=model, data=data, sparse=sparse)

    correction_matrix_dict = {}
    if sparse:
        correction_matrix_dict[0] = (
            torch.eye(data.num_nodes).to_sparse().to(data.x.device)
        )
        for idx in range(1, num_hops):
            correction_matrix_dict[idx] = torch.sparse.mm(
                correction_matrix_dict[idx - 1], att_matrix_dict[num_hops - idx]
            )
    else:
        correction_matrix_dict[0] = torch.eye(data.num_nodes).to(data.x.device)
        for idx in range(1, num_hops):
            correction_matrix_dict[idx] = (
                correction_matrix_dict[idx - 1] @ att_matrix_dict[num_hops - idx]
            )

    return att_matrix_dict, correction_matrix_dict


def avgatt(
    target_edge: Tuple[int, int],
    att_matrix_dict: Dict,
    num_hops: int,
) -> float:
    """
    Calculates AvgAtt for a given target edge.
    This is the average of attention values across all layers.
    Args:
        target_edge: The target edge.
        att_matrix_dict: The dictionary of attention matrices.
        num_hops: The number of layers of the GAT model.
    Returns:
        The AvgAtt value.
    """
    src_idx = target_edge[0]
    tgt_idx = target_edge[1]

    result = 0
    for m in range(num_hops):
        result += att_matrix_dict[m][tgt_idx, src_idx]
    return result.item() / num_hops


def gatt(
    target_edge: Tuple[int, int],
    ref_node: int,
    att_matrix_dict: Dict,
    correction_matrix_dict: Dict,
    num_hops=int,
) -> float:
    """
    Calculates GAtt for a given target edge in the context of calculating the reference node.
    In the paper, GAtt is expressed as \phi_{i, j}^v. Target edge is (i, j) and reference node is v.
    Args:
        target_edge: The target edge.
        ref_node: The reference node.
        att_matrix_dict: The dictionary of attention matrices.
        correction_matrix_dict: The dictionary of correction matrices.
    Returns:
        The GAtt value.
    """
    src_idx = target_edge[0]
    tgt_idx = target_edge[1]

    assert num_hops > 1, "1 layer models are out of the question."

    result = 0

    for m in range(num_hops - 1):
        result += (
            correction_matrix_dict[num_hops - m - 1][ref_node, tgt_idx].item()
            * att_matrix_dict[m][tgt_idx, src_idx]
        )
    if tgt_idx == ref_node:
        result += att_matrix_dict[num_hops - 1][tgt_idx, src_idx]
    return result.item()


def get_gatt(
    target_node,
    model,
    data,
    sparse: bool = True,
) -> Tuple[List[float], List[Tuple[int, int]]]:
    num_hops = get_num_hops(model=model)
    att_matrix_dict, correction_matrix_dict = prep_for_gatt(
        model=model, data=data, num_hops=num_hops, sparse=sparse
    )

    edges_in_k_hop = return_edges_in_k_hop(
        data=data, target_idx=target_node, hop=num_hops, self_loops=True
    )

    gatt_list = []
    for current_edge in edges_in_k_hop:
        gatt_list.append(
            gatt(
                target_edge=current_edge,
                ref_node=target_node,
                att_matrix_dict=att_matrix_dict,
                correction_matrix_dict=correction_matrix_dict,
                num_hops=num_hops,
            )
        )

    return gatt_list, edges_in_k_hop


def get_avgatt(
    target_node,
    model,
    data,
    sparse: bool = True,
) -> Tuple[List[float], List[Tuple[int, int]]]:
    num_hops = get_num_hops(model=model)
    att_matrix_dict, _ = prep_for_gatt(
        model=model, data=data, num_hops=num_hops, sparse=sparse
    )  # correction_matrix_dict is not needed for avgatt

    edges_in_k_hop = return_edges_in_k_hop(
        data=data, target_idx=target_node, hop=num_hops, self_loops=True
    )

    avgatt_list = []
    for current_edge in edges_in_k_hop:
        avgatt_list.append(
            avgatt(
                target_edge=current_edge,
                att_matrix_dict=att_matrix_dict,
                num_hops=num_hops,
            )
        )

    return avgatt_list, edges_in_k_hop


def gatt_batch(
    ref_node: int,
    num_of_hops: int,
    att_matrix_dict: Dict,
    correction_matrix_dict: Dict,
    sparse: bool = False,
) -> torch.Tensor:
    num_nodes = att_matrix_dict[0].shape[0]
    # Set the select matrix
    if sparse:
        indices = torch.tensor(
            [
                [ref_node] * att_matrix_dict[0].shape[1],
                list(range(att_matrix_dict[0].shape[1])),
            ]
        )
        values = torch.ones(att_matrix_dict[0].shape[1])
        select_matrix = torch.sparse_coo_tensor(
            indices, values, att_matrix_dict[0].shape, device=att_matrix_dict[0].device
        )
    else:
        select_matrix = torch.zeros_like(att_matrix_dict[0])
        select_matrix[ref_node, :] = 1

    # Loop over the number of hops
    for i in reversed(range(num_of_hops)):
        # If it's the hightest hop, use the select matrix
        if i == num_of_hops - 1:
            if sparse:
                result_matrix = select_matrix * att_matrix_dict[i]
            else:
                result_matrix = select_matrix * att_matrix_dict[i]
        # If it's not the highest hop, use the correction matrix if it's provided
        else:
            if sparse:
                # Cannot use index operation for sparse tensors
                row_selector = torch.sparse_coo_tensor(
                    torch.tensor([[0], [ref_node]]),
                    torch.tensor([1.0]),
                    (1, num_nodes),
                    device=att_matrix_dict[0].device,
                )
                selected = torch.sparse.mm(
                    row_selector, correction_matrix_dict[num_of_hops - i - 1]
                )
                # Identical operation as .expand_as but in sparse format
                expanded = torch.vstack([selected] * att_matrix_dict[i].shape[1]).t()

                result_matrix += expanded * att_matrix_dict[i]
            else:
                result_matrix += (
                    correction_matrix_dict[num_of_hops - i - 1][ref_node, :]
                    .expand_as(att_matrix_dict[i])
                    .t()
                    * att_matrix_dict[i]
                )

    return result_matrix


def get_gatt_batch(
    target_node,
    model,
    data,
    sparse: bool = True,
) -> Tuple[List[float], List[Tuple[int, int]]]:
    num_hops = get_num_hops(model=model)
    att_matrix_dict, correction_matrix_dict = prep_for_gatt(
        model=model, data=data, num_hops=num_hops, sparse=sparse
    )

    edges_in_k_hop = return_edges_in_k_hop(
        data=data,
        target_idx=target_node,
        hop=num_hops,
        self_loops=True,
        return_as_tensor=True,
    )

    gatt_matrix = gatt_batch(
        ref_node=target_node,
        num_of_hops=num_hops,
        att_matrix_dict=att_matrix_dict,
        correction_matrix_dict=correction_matrix_dict,
        sparse=sparse,
    )

    if sparse:
        gatt_list = [
            gatt_matrix[edge[1], edge[0]].item() for edge in edges_in_k_hop.t()
        ]
    else:
        gatt_list = gatt_matrix[edges_in_k_hop[1], edges_in_k_hop[0]].tolist()

    return gatt_list, edges_in_k_hop

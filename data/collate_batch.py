# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch


def train_collate_fn(batch):
    imgs, pids, _, _, = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids

def val_collate_fn(batch):
    imgs, pids, camids, _ = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids

def train_collate_graph_fn(batch):
    samples, pids, _, _, = zip(*batch)
    samples = [data for data in samples]
    pids = torch.tensor(pids, dtype=torch.int64)
    return samples, pids

def val_collate_graph_fn(batch):
    samples, pids, camids, _, = zip(*batch)
    samples = [data for data in samples] 
    pids = torch.tensor(pids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return samples, pids, camids

def train_combined_collate_fn(batch):
    imgs, pids, _, _, samples = zip(*batch)
    imgs = torch.stack(imgs, dim=0)  # Ensure imgs are tensors
    pids = torch.tensor(pids, dtype=torch.int64)  # Convert pids to a tensor
    return imgs, pids, samples

def val_combined_collate_fn(batch):
    imgs, pids, camids, _, samples = zip(*batch)
    imgs = torch.stack(imgs, dim=0)  # Ensure imgs are tensors
    pids = torch.tensor(pids, dtype=torch.int64)  # Convert pids to a tensor
    camids = torch.tensor(camids, dtype=torch.int64)  # Convert camids to a tensor
    return imgs, pids, camids, samples

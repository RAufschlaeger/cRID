# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch.utils.data import DataLoader
import torch

from .collate_batch import train_collate_fn, val_collate_fn, train_collate_graph_fn, val_collate_graph_fn
from .datasets import init_dataset, ImageDataset
from data.src.data.graph_dataset import GraphDataset
from .samplers import RandomIdentitySampler, RandomIdentitySampler_alignedreid  # New add by gu
from .transforms import build_transforms


def make_data_loader(cfg):
    train_transforms = build_transforms(cfg, is_train=True)
    val_transforms = build_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    dataset_types = getattr(cfg.DATASETS, "TYPES", ("image",))  # Default to ("image",) if TYPES is not defined

    dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)

    num_classes = dataset.num_train_pids
    train_loader_image, train_loader_graph = None, None

    if "image" in dataset_types:
        train_set = ImageDataset(dataset.train, train_transforms)
        if cfg.DATALOADER.SAMPLER == 'softmax':
            train_loader_image = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
                collate_fn=train_collate_fn
            )
        else:
            train_loader_image = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_fn
            )
        val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
        val_loader_image = DataLoader(val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn)

    if "graph" in dataset_types:
        
        train_set_path = './data/graphs/Market-1501-v15.09.15/bounding_box_train/graph_dataset_allenai-Molmo-7B-O-0924.pth'
        train_set = torch.load(train_set_path)

        query_set_path = './data/graphs/Market-1501-v15.09.15/query/graph_dataset_allenai-Molmo-7B-O-0924.pth'
        query_set = torch.load(query_set_path)

        gallery_set_path = './data/graphs/Market-1501-v15.09.15/bounding_box_test/graph_dataset_allenai-Molmo-7B-O-0924.pth'
        gallery_set = torch.load(gallery_set_path)

        # Relabel PIDs for train_set
        train_pid_container = set(data[1].item() for data in train_set)
        train_pid2label = {pid: label for label, pid in enumerate(train_pid_container)}

        for i, data in enumerate(train_set):
            pid = data[1].item()
            relabeled_pid = train_pid2label[pid]
            train_set[i] = (data[0], relabeled_pid, *data[2:])

        # Update: No relabel! ...
        # Relabel PIDs for query_set and gallery_set together
        query_gallery_pid_container = set(data[1].item() for data in (query_set + gallery_set))
        query_gallery_pid2label = {pid: label for label, pid in enumerate(query_gallery_pid_container)}

        def relabel_query_gallery(dataset):
            for i, data in enumerate(dataset):
                pid = data[1].item()
                relabeled_pid = query_gallery_pid2label[pid]
                dataset[i] = (data[0], pid, *data[2:])
            return dataset

        query_set = relabel_query_gallery(query_set)
        gallery_set = relabel_query_gallery(gallery_set)

        val_set = query_set + gallery_set
        val_loader_graph = DataLoader(val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_graph_fn)


        if cfg.DATALOADER.SAMPLER == 'softmax':
            train_loader_graph = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
                collate_fn=train_collate_graph_fn
            )
        else:
            train_loader_graph = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                # sampler=RandomIdentitySampler_alignedreid(train_set, cfg.DATALOADER.NUM_INSTANCE),
                sampler=RandomIdentitySampler(train_set, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_graph_fn
            )

    if "image" in dataset_types and "graph" in dataset_types:
        return train_loader_image, train_loader_graph, val_loader_image, val_loader_graph, len(dataset.query), num_classes
    elif "image" in dataset_types:
        return train_loader_image, val_loader_image, len(dataset.query), num_classes
    elif "graph" in dataset_types:
        return train_loader_graph, val_loader_graph, len(dataset.query), num_classes
    else:
        raise ValueError("Unsupported dataset type in cfg.DATASETS.TYPES")

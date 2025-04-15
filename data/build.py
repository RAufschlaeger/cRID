# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch.utils.data import DataLoader
import torch

from .collate_batch import train_collate_fn, val_collate_fn, train_collate_graph_fn, val_collate_graph_fn, train_combined_collate_fn, val_combined_collate_fn
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

    # # Sort image datasets
    # dataset.train = sorted(dataset.train, key=lambda x: x[0])
    # dataset.gallery = sorted(dataset.gallery, key=lambda x: x[0])
    # dataset.query = sorted(dataset.query, key=lambda x: x[0])
    
    if "image" in dataset_types and "graph" not in dataset_types:
        train_set_image = ImageDataset(dataset.train, train_transforms)
        if cfg.DATALOADER.SAMPLER == 'softmax':
            train_loader_image = DataLoader(
                train_set_image, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
                collate_fn=train_collate_fn
            )
        else:
            train_loader_image = DataLoader(
                train_set_image, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_fn
            )
        val_set_image = ImageDataset(dataset.query + dataset.gallery, val_transforms)
        val_loader_image = DataLoader(val_set_image, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn)

    elif "image" not in dataset_types and "graph" in dataset_types:
        
        train_set_path = './data/graphs/Market-1501-v15.09.15/bounding_box_train/graph_dataset_allenai-Molmo-7B-O-0924.pth'
        train_set_graph = torch.load(train_set_path)

        query_set_path = './data/graphs/Market-1501-v15.09.15/query/graph_dataset_allenai-Molmo-7B-O-0924.pth'
        query_set_graph = torch.load(query_set_path)

        gallery_set_path = './data/graphs/Market-1501-v15.09.15/bounding_box_test/graph_dataset_allenai-Molmo-7B-O-0924.pth'
        gallery_set_graph = torch.load(gallery_set_path)

        # Relabel PIDs for train_set
        train_pid_container = set(data[1].item() for data in train_set_graph)
        train_pid2label = {pid: label for label, pid in enumerate(train_pid_container)}

        for i, data in enumerate(train_set_graph):
            pid = data[1].item()
            relabeled_pid = train_pid2label[pid]
            train_set_graph[i] = (data[0], relabeled_pid, *data[2:])

        # Update: No relabel! ...

        def tensor_to_int(dataset):
            for i, data in enumerate(dataset):
                pid = data[1].item()
                dataset[i] = (data[0], pid, *data[2:])
            return dataset  

        # train_set_graph = rm_tensor(train_set_graph)
        query_set_graph = tensor_to_int(query_set_graph)
        gallery_set_graph = tensor_to_int(gallery_set_graph)

        val_set_graph = query_set_graph + gallery_set_graph
        val_loader_graph = DataLoader(val_set_graph, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_graph_fn)

        if cfg.DATALOADER.SAMPLER == 'softmax':
            train_loader_graph = DataLoader(
                train_set_graph, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
                collate_fn=train_collate_graph_fn
            )
        else:
            train_loader_graph = DataLoader(
                train_set_graph, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                # sampler=RandomIdentitySampler_alignedreid(train_set, cfg.DATALOADER.NUM_INSTANCE),
                sampler=RandomIdentitySampler(train_set_graph, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_graph_fn
            )
            
    elif "image" in dataset_types and "graph" in dataset_types:

        train_set_path = './data/graphs/Market-1501-v15.09.15/bounding_box_train/graph_dataset_allenai-Molmo-7B-O-0924.pth'
        train_set_graph = torch.load(train_set_path)

        query_set_path = './data/graphs/Market-1501-v15.09.15/query/graph_dataset_allenai-Molmo-7B-O-0924.pth'
        query_set_graph = torch.load(query_set_path)

        gallery_set_path = './data/graphs/Market-1501-v15.09.15/bounding_box_test/graph_dataset_allenai-Molmo-7B-O-0924.pth'
        gallery_set_graph = torch.load(gallery_set_path)

        # Replace id with id.item() in graph datasets
        train_set_graph = [(data[0], data[1].item(), data[2].item(), *data[3:]) for data in train_set_graph]
        gallery_set_graph = [(data[0], data[1].item(), data[2].item(), *data[3:]) for data in gallery_set_graph]
        query_set_graph = [(data[0], data[1].item(), data[2].item(), *data[3:]) for data in query_set_graph]

        # Sort graph datasets
        train_set_graph = sorted(train_set_graph, key=lambda x: x[3])
        gallery_set_graph = sorted(gallery_set_graph, key=lambda x: x[3])
        query_set_graph = sorted(query_set_graph, key=lambda x: x[3])

        # Sort image datasets
        dataset.train = sorted(dataset.train, key=lambda x: x[0])
        dataset.gallery = sorted(dataset.gallery, key=lambda x: x[0])
        dataset.query = sorted(dataset.query, key=lambda x: x[0])

        train_set_image = ImageDataset(dataset.train, None)
        gallery_set_image = ImageDataset(dataset.gallery, None)
        query_set_image = ImageDataset(dataset.query, None)

        # Ensure both datasets are aligned
        assert len(train_set_image) == len(train_set_graph), "Image and graph datasets must have the same length."

        # Extract specific indices from images and graphs
        combined_train_set = [
            (img[0], img[1], img[2], img[3], graph[0]) 
            for img, graph in zip(train_set_image, train_set_graph)
        ]
        combined_val_set = [
            (img[0], img[1], img[2], img[3], graph[0]) 
            for img, graph in zip(query_set_image + gallery_set_image, query_set_graph + gallery_set_graph)
        ]

        if cfg.DATALOADER.SAMPLER == 'softmax':
            train_loader_combined = DataLoader(
                combined_train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
                collate_fn=train_combined_collate_fn
            )
        else:
            sampler = RandomIdentitySampler(
                combined_train_set, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE
            )
            train_loader_combined = DataLoader(
                combined_train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, sampler=sampler, num_workers=num_workers,
                collate_fn=train_combined_collate_fn
            )

        val_loader_combined = DataLoader(
            combined_val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
            collate_fn=val_combined_collate_fn
        )

        return train_loader_combined, val_loader_combined, len(dataset.query), num_classes
    
    else:
        raise ValueError("Unsupported dataset type in cfg.DATASETS.TYPES")

    if "image" in dataset_types:
        return train_loader_image, val_loader_image, len(dataset.query), num_classes
    elif "graph" in dataset_types:
        return train_loader_graph, val_loader_graph, len(dataset.query), num_classes
    else:
        raise ValueError("Unsupported dataset type in cfg.DATASETS.TYPES")

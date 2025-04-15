# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .baseline import Baseline
from .extbaseline import ExtBaseline
from .graph_transformer import GraphTransformer


def build_model(cfg, num_classes):
    dataset_types = getattr(cfg.DATASETS, "TYPES", ("image",))  # Default to ("image",) if TYPES is not defined

    if 'image' in dataset_types and 'graph' in dataset_types:
        in_channels = cfg.GRAPH.IN_CHANNELS
        out_features = cfg.GRAPH.OUT_FEATURES
        model = ExtBaseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE
                            , in_channels, out_features)
        return model
    elif 'image' in dataset_types:
        model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE)
        return model
    elif 'graph' in dataset_types:
        in_channels = cfg.GRAPH.IN_CHANNELS
        out_features = cfg.GRAPH.OUT_FEATURES
        model = GraphTransformer(num_classes, in_channels, out_features)
        return model

    else:
        raise ValueError("Unsupported DATASETS.TYPES. Supported types are 'image', 'graph', and 'image+graph'.")

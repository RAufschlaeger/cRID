# encoding: utf-8
"""
@author:  raufschlaeger
"""

import os
import sys
from datetime import datetime

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from data.src.models.molmo7b import Molmo
from inference.utils import extract_node_strings, visualize_gatt

import os
import sys
import shutil

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import numpy as np

import networkx as nx
import torch
import pandas as pd
from matplotlib import pyplot as plt

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


image_path = "/home/raufschlaeger/reid-strong-baseline/data/market1501/bounding_box_test/0473_c5s1_130020_01.jpg"
extractor = Molmo()

graph = extractor.process_image(image_path)

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
# missing = model.load_state_dict(teacher_state_no_gat, strict=False)
# print("Missing keys:", missing.missing_keys)
# print("Unexpected keys:", missing.unexpected_keys)

# Print only the keys that were successfully loaded (i.e., present in both model and teacher_state_no_gat)
# loaded_keys = [k for k in teacher_state_no_gat.keys() if k in model.state_dict().keys()]
# print("Successfully loaded keys:", loaded_keys)

# Extract node strings
node_strings = extract_node_strings(graph)
print("Extracted Node Strings:", node_strings)

image_paths = [
    "./data/market1501/query/0473_c4s2_050698_00.jpg",
    "./data/market1501/bounding_box_test/0473_c4s2_050723_01.jpg",
    "./data/market1501/bounding_box_test/0473_c1s2_050246_02.jpg",
    "./data/market1501/bounding_box_test/0473_c4s2_050748_01.jpg",
    "./data/market1501/bounding_box_test/0473_c4s2_050698_01.jpg",
    "./data/market1501/bounding_box_test/0473_c5s1_130020_01.jpg"
]

for image_path in image_paths:
    print(f"Processing {image_path}")

    # Copy original jpg to inference/test_query
    base_name = os.path.basename(image_path)
    dest_path = os.path.join("./inference/test_query/", base_name)
    shutil.copy2(image_path, dest_path)

    graph = extractor.process_image(image_path)
    node_strings = extract_node_strings(graph)
    print("Extracted Node Strings:", node_strings)

    visualize_gatt(model, graph, node_strings, base_name)

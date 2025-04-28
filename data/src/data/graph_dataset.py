# encoding: utf-8
"""
@author:  raufschlaeger
"""

import pandas as pd
import torch
import torch_geometric.data
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer

from ..processing.scene_graph import SceneGraph

from data.src.models.phi4 import fix_graph
from data.src.processing.scene_graph import fix_malformed_json

# import fasttext
# from huggingface_hub import hf_hub_download

# def get_fasttext_embedding(self, word):
#     return self.ft_model.get_word_vector(word)


class GraphDataset(torch_geometric.data.Dataset):
    def __init__(self, annotations_file):
        """
        Initializes the dataset with the provided annotations file and image directory.

        Args:
            annotations_file (str): Path to the CSV file containing image names and labels.
        """
        super().__init__()
        self.annotations_file = annotations_file
        self.annotations_df = pd.read_csv(annotations_file)
        self.annotations_df["fixed_graph"] = self.annotations_df["fixed_graph"].astype(str)  # Explicitly cast column to string
        self.graph_labels = pd.read_csv(annotations_file)  # Load the annotations CSV
        self.st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def __len__(self):
        """
        Returns the length of the dataset (i.e., the number of images).
        """
        return len(self.graph_labels)

    def __getitem__(self, idx):
        return self.process_single(idx)

    def process_single(self, idx):

        graph = self.graph_labels.iloc[idx, -2]  # assumes that graph is in second last column

        try:
            scene_graph = SceneGraph(graph, self.st_model)
            sample = scene_graph.nx_to_tg_data()
            fixed_graph = None

        except Exception as e:
            print(f"1st ERROR: {str(e)}\n")
            print("malformed graph: ", graph)
            # fix malformed JSON
            fixed_graph = fix_graph(graph)
            fixed_graph = fix_malformed_json(fixed_graph)
            print("fixed graph: ", fixed_graph)

            try:
                scene_graph = SceneGraph(fixed_graph, self.st_model)
                sample = scene_graph.nx_to_tg_data()
            except Exception as e:
                print(f"2nd ERROR ...: {str(e)}\n")
                print("malformed graph: ", graph)
                fixed_graph = fix_graph(fixed_graph)
                fixed_graph = fix_malformed_json(fixed_graph)
                print("fixed graph: ", fixed_graph)
                scene_graph = SceneGraph(graph, self.st_model)
                sample = scene_graph.nx_to_tg_data()

        if isinstance(sample, Data):
            pid = torch.tensor(self.graph_labels.iloc[idx, 1], dtype=torch.long)
            camid = torch.tensor(self.graph_labels.iloc[idx, 2], dtype=torch.long)
            img_path = self.graph_labels.iloc[idx, 0]

            if fixed_graph is not None:
                self.annotations_df.loc[idx, "fixed_graph"] = str(fixed_graph)  # Explicitly cast to string
            
                # Save updated annotations file
                self.annotations_df.to_csv(self.annotations_file, index=False)

            return sample, pid, camid, img_path
        else:
            raise ValueError("The output of nx_to_tg_data() is not a Data object.")

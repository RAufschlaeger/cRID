# Add the project root directory to the Python path
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
print(f"Project root: {project_root}")
sys.path.insert(0, project_root)

import time

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from PIL import Image
from sentence_transformers import SentenceTransformer

from data.src.processing.scene_graph import SceneGraph


def plot_graph(G, graph, save_path=None):
    """
    Plots the given NetworkX graph with attributes and relationships.

    Args:
        G (nx.DiGraph): Graph to plot.
        graph (str): graph.
        save_path (str, optional): File path to save the plot. Defaults to None.
    """
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    node_colors = [G.nodes[node].get('color', 'skyblue') for node in G.nodes()]
    nx.draw(G, pos, with_labels=True, node_size=800, node_color=node_colors, font_size=10, arrows=True, arrowsize=10)
    edge_labels = nx.get_edge_attributes(G, 'relationship')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.text(0.5, 1.05, graph, horizontalalignment='center', transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(facecolor='white', alpha=0.7))
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Graph saved to {save_path}")
    else:
        plt.show()


# TODO: adjust for existing img and annotations files


if __name__ == "__main__":
    start_time = time.time()

    # image-based:
    # from ...src.models.vlm_extractor import VLMExtractor
    #
    # extractor = VLMExtractor(model_name='allenai/Molmo-7B-D-0924')  # google/paligemma2-3b-pt-224'

    split = "bounding_box_train"

    # img_name = "1273_c5s3_026340_00.jpg"
    img_name = "0977_c1s4_062936_02.jpg"
    img_name = "0612_c6s2_031593_01.jpg"
    img_path = "./data/raw/market1501/" + split + "/" + img_name

    print(f"Image: {img_name}")

    #TODO: specify file
    annotations_path = "./data/annotations/Market-1501-v15.09.15/" + split + "/annotations_allenai-Molmo-7B-O-0924.csv"

    annotations = pd.read_csv(annotations_path)

    try:
        # # Load the image
        # image = Image.open(img_path).convert("RGB")
        #
        # # Process the image
        # print(f"Processing Image {img_name}...")
        # caption = extractor.process_image(image)

        # Filter the DataFrame to get the matching caption
        matching_row = annotations[annotations['filename'] == img_name]

        # Extract the graph
        graph = matching_row.iloc[0]['graph']
        print("Graph: ", graph)

        # Generate the scene graph
        scene_graph = SceneGraph(graph, SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'))

        # Plot and save the graph
        plot_graph(scene_graph.G, graph, save_path="./graph.png")
    except FileNotFoundError:
        print(f"Image file not found: {img_path}")
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time (s): " + str(execution_time))


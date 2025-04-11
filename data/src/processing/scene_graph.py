import json
import networkx as nx
import torch
import numpy as np
from torch_geometric.data import Data
import re


def clean_json_string(text):
    parts = text.split("```")
    if len(parts) > 1:
        content = parts[1]  # Split on closing marker
        return content.strip("json")
    else:
        return text


def fix_malformed_json(json_str):
    """
    Attempts to fix malformed JSON by correcting common structural issues.

    This function performs the following fixes:
    1. **Extract JSON block**: Isolates content from the first `{` to the last `}`.
    2. **Remove Markdown-style code block markers**: Removes ```json and ``` if present.
    3. **Check if JSON is valid**: If valid, return formatted.
    4. **Fix double-double quotes**: Replaces `""` with `"` to correct improper escaping.
    5. **Convert malformed edge sets**: Fixes `{ "source", "target", "relation" }` style.
    6. **Remove trailing commas**: Cleans up `,]` and `,}`.
    7. **Return parsed and formatted JSON or error**.

    Parameters:
        json_str (str): The input JSON string, which may be malformed or surrounded by text.

    Returns:
        str: A corrected JSON string formatted with indentation, or an error message.
    """
    # Step 1: Extract content from the first "{" to the last "}"
    match = re.search(r'\{.*\}', json_str, re.DOTALL)
    if not match:
        return "Error: No valid JSON object found in input."
    json_str = match.group(0)

    # Step 2: Remove Markdown-style code block markers
    json_str = re.sub(r'^```(?:json)?\s*', '', json_str.strip(), flags=re.IGNORECASE | re.MULTILINE)
    json_str = re.sub(r'\s*```$', '', json_str.strip(), flags=re.MULTILINE)

    # Step 3: Try parsing directly
    try:
        return json.dumps(json.loads(json_str), indent=2)
    except json.JSONDecodeError:
        pass  # Proceed with fixing

    # Step 4: Fix double-double quotes ("" -> ")
    json_str = json_str.replace('""', '"')

    # Step 5: Convert malformed edge sets
    malformed_edges = re.findall(r'\{\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\}', json_str)
    for source, target, relation in malformed_edges:
        correct_edge = f'{{"source": "{source}", "target": "{target}", "relation": "{relation}"}}'
        json_str = re.sub(
            r'\{\s*"' + re.escape(source) + r'"\s*,\s*"' + re.escape(target) + r'"\s*,\s*"' + re.escape(relation) + r'"\s*\}',
            correct_edge,
            json_str,
            count=1
        )

    # Step 6: Remove trailing commas
    json_str = re.sub(r',\s*([\]}])', r'\1', json_str)

    # Step 7: Final parse attempt
    try:
        data = json.loads(json_str)
        return json.dumps(data, indent=2)
    except json.JSONDecodeError as e:
        return f"Error fixing JSON: {e}"

    

class SceneGraph:
    def __init__(self, graph: str, st_model):

        json_data = clean_json_string(graph)

        try:
            # Attempt to parse the JSON
            graph = json.loads(json_data)
        except json.JSONDecodeError as e:
            graph = fix_malformed_json(json_data)

        # Now proceed with normal initialization
        self.nodes = graph.get("nodes", [])
        self.edges = graph.get("edges", [])

        self.G = nx.DiGraph()
        self._build_graph()

        # Load Sentence Transformer model
        self.st_model = st_model

    def get_st_embedding(self, text):
        # The model expects a list of sentences; hence, [text]
        embedding = self.st_model.encode([text])
        return embedding[0]  # Return the embedding of the single input text

    def _build_graph(self):
        for node in self.nodes:
            self.G.add_node(node['id'], color='skyblue')

            # Use .get() to avoid KeyError if 'attributes' is missing
            attributes = node.get('attributes', [])
            if isinstance(attributes, list):
                for attr in attributes:
                    if isinstance(attr, dict):
                        attr_str = ' '.join(str(value) for value in attr.values())
                        self.G.add_node(attr_str, color='red')
                        self.G.add_edge(node['id'], attr_str, relationship='has attribute')
                    else:
                        self.G.add_node(attr, color='red')
                        self.G.add_edge(node['id'], attr, relationship='has attribute')
            elif isinstance(attributes, dict):
                # Handle single dictionary as attributes
                attr_str = ' '.join(str(value) for value in attributes.values())
                self.G.add_node(attr_str, color='red')
                self.G.add_edge(node['id'], attr_str, relationship='has attribute')

        for edge in self.edges:
            if isinstance(edge, dict) and 'source' in edge and 'target' in edge:
                self.G.add_edge(
                    edge['source'],
                    edge['target'],
                    relationship=edge.get("relation", "unknown")
                )
            elif isinstance(edge, (set, list)):
                edge_list = list(edge)
                if len(edge_list) == 3:
                    print(f"⚠️ Warning: Fixing malformed edge: {edge_list}")
                    self.G.add_edge(
                        edge_list[0],
                        edge_list[1],
                        relationship=edge_list[2]
                    )
                else:
                    print(f"❌ Skipping invalid edge: {edge}")

    def nx_to_tg_data(self):
        nodes = list(self.G.nodes)
        node_mapping = {node: idx for idx, node in enumerate(nodes)}
        edge_index = []
        edge_attr = []
        node_embeddings = [self.get_st_embedding(node) for node in nodes]

        for u, v, data in self.G.edges(data=True):
            edge_index.append([node_mapping[u], node_mapping[v]])
            edge_attr.append(self.get_st_embedding(data.get('relationship', '')))

        x = torch.tensor(np.array(node_embeddings), dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

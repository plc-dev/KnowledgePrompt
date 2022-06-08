from google.colab import drive
import os
import json
from typing import Any, Dict
import networkx as nx
import matplotlib.pyplot as plt
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
from typing import List

class PersistenceLayer:
    def __init__(self):
        drive_path = "/content/drive"
        drive.mount(drive_path)
        self.base_directory = f"{drive_path}/My Drive/Prompts"

    def persist_dict(self, data: Dict[str, Any]) -> None:
        file_path = self.base_directory
        with open(file_path, "w", encoding="utf8") as f:
            json.dump(data, f)

    # /content/gdrive/My Drive/Output_folder
    def __create_folder(self, folder_path: str) -> None:
        try:
            os.mkdir(folder_path)
        except:
            print("Folder already exists")

def digraph_from_synset_with_function(synset: object, fn) -> nx.DiGraph:
    seen = set()
    graph = nx.DiGraph()

    def recurse(s):
        if not s in seen:
            seen.add(s)
            graph.add_node(s.lemma_names()[0])
            for s1 in fn(s):
                graph.add_node(s1.lemma_names()[0])
                graph.add_edge(s1.lemma_names()[0], s.lemma_names()[0])
                recurse(s1)

    recurse(synset)
    return graph


def graph_draw(
    graph: nx.DiGraph, 
    fig_size: int = 12, 
    show_labels: bool = True, 
    layout_algorithm: str = "dot"
    ) -> None:
    pos = graphviz_layout(graph, prog=layout_algorithm)
    plt.figure(3, figsize=(fig_size, fig_size))
    nx.draw(graph, pos, with_labels=show_labels)
    plt.show()

import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import pathlib
import time
import graphviz
import h5py


def visualize_graph(graph, color='red', name="default"):
    G_nx = nx.Graph()
    colors = []
    for vertex in graph.vertex_list:
        colors.append("red" if vertex.coarse else "blue")

    for vertex in graph.vertex_list:
        G_nx.add_node(vertex.id)

    adj_matrix = graph.vertex_list_to_adj_matrix(graph.vertex_list)

    u_indices, v_indices = np.nonzero(adj_matrix)
    for u_idx, v_idx in zip(u_indices, v_indices):
        vertex_u = list(graph.vertex_list)[u_idx]
        vertex_v = list(graph.vertex_list)[v_idx]
        u = vertex_u.id
        v = vertex_v.id
        if u != v:
            G_nx.add_edge(u, v)
    
    pos = nx.kamada_kawai_layout(G_nx)
    plt.figure(figsize=(8, 6))
    nx.draw(G_nx, pos, with_labels=True, node_color=colors, node_size=500, font_weight='bold')
    plt.title("Graph")
    plt.savefig(f"images/{name}.png", dpi=300, bbox_inches='tight')
    plt.close()

def visualize_graph_2(graph, name):
    g = graphviz.Graph(name=name, format='png') 
    for vertex in graph.vertex_list:
        g.node(f"{vertex.id}")
        for neighbour, weight in vertex.adj:
            g.edge(f"{vertex.id}", f"{neighbour.id}", f"{weight}")
    g.render(format='png')

def generate_grid_graph(rows, cols, filename, type="csv"):
    G = nx.grid_2d_graph(rows, cols)
    
    A = nx.to_scipy_sparse_array(G, format="coo", dtype=int)

    dataframe = pd.DataFrame({
        "row": A.row,
        "col": A.col,
        "val": A.data
    })
    
    if type == "csv":
        dataframe.to_csv(filename, index=False)
    elif type == "hdf5":
        with h5py.File(filename, mode="w") as file:
            group = file.require_group("coo_matrix")
            group.create_dataset("row", data=dataframe["row"].to_numpy())
            group.create_dataset("col", data=dataframe["col"].to_numpy())
            group.create_dataset("val", data=dataframe["val"].to_numpy(dtype=np.float64))

def clear_folder_or_create(folder_path):
    folder = pathlib.Path(folder_path)

    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)

    for filename in folder.iterdir():
        filename.unlink()
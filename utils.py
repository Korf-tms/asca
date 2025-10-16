import networkx as nx
import matplotlib.pyplot as plt
import random
import csv
import numpy as np

#utils are all ai generated, will change someday

def visualize_graph(graph, color='red'):
    G_nx = nx.Graph()
    colors = []
    for vertex in graph.vertex_list:
        try:
            colors.append("red" if vertex.original_vertex.coarse else "blue")
        except AttributeError:
            colors.append("red" if vertex.coarse else "blue")

    for vertex in graph.vertex_list:
        try:
            G_nx.add_node(vertex.original_vertex.id)
        except AttributeError:
            G_nx.add_node(vertex.id)

    adj_matrix = graph.vertex_list_to_adj_matrix(graph.vertex_list)

    u_indices, v_indices = np.nonzero(adj_matrix)
    for u_idx, v_idx in zip(u_indices, v_indices):
        vertex_u = list(graph.vertex_list)[u_idx]
        vertex_v = list(graph.vertex_list)[v_idx]
        try:
            u = vertex_u.original_vertex.id
            v = vertex_v.original_vertex.id
        except AttributeError:
            u = vertex_u.id
            v = vertex_v.id
        if u != v:
            G_nx.add_edge(u, v)

    pos = nx.kamada_kawai_layout(G_nx)
    plt.figure(figsize=(8, 6))
    nx.draw(G_nx, pos, with_labels=True, node_color=colors, node_size=500, font_weight='bold')
    plt.title("Graph")
    plt.savefig(f"images/{graph.name}.png", dpi=300, bbox_inches='tight')
    plt.close()



def generate_graph_to_coo_csv(rows, cols, csv_filename, connection_prob=0.9):
    """
    Generate a perturbed grid-like graph, ensure connectivity,
    and export its adjacency matrix in COO sparse format to CSV.
    
    Args:
        rows (int): Number of rows in grid
        cols (int): Number of cols in grid
        csv_filename (str): Output CSV file path
        connection_prob (float): Probability to keep an edge
    """
    # Start with grid
    G = nx.grid_2d_graph(rows, cols)

    edges = list(G.edges())
    random.shuffle(edges)
    
    # Randomly remove edges
    for u, v in edges:
        if random.random() > connection_prob:
            if G.degree(u) > 1 and G.degree(v) > 1:
                G.remove_edge(u, v)
    
    # Ensure graph is connected
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        while len(components) > 1:
            c1, c2 = components[0], components[1]
            min_dist = float('inf')
            best_pair = (None, None)
            
            for u in c1:
                for v in c2:
                    i1, j1 = u
                    i2, j2 = v
                    dist = abs(i1-i2) + abs(j1-j2)
                    if dist < min_dist:
                        min_dist = dist
                        best_pair = (u, v)
            
            G.add_edge(*best_pair)
            components = list(nx.connected_components(G))
    
    # Relabel nodes to integers
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)

    # Convert to sparse adjacency matrix (COO)
    A = nx.to_scipy_sparse_array(G, format="coo", dtype=int)

    # Save COO format (row, col, value) to CSV
    with open(csv_filename, mode="w", newline="") as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(["row", "col", "val"])
        for r, c, v in zip(A.row, A.col, A.data):
            writer.writerow([r, c, int(v)])  # adjacency is 1s
    
    return G, A

import os
import shutil

def clear_folder(folder_path):
    """
    Deletes all files and subdirectories in the given folder.
    
    Parameters:
        folder_path (str): The path to the folder to clear.
    """
    if not os.path.exists(folder_path):
        print(f"Folder does not exist: {folder_path}")
        return

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # delete file or symbolic link
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # delete folder recursively
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
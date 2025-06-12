import networkx as nx
import matplotlib.pyplot as plt
from graph import GraphVertex
import random


def visualize_graph(graph, color='red'):
    G_nx = nx.Graph()

    for vertex in graph.vertex_dict.values():
        G_nx.add_node(vertex.id)
            
    coo = graph.adjacency_matrix
    for u_idx, v_idx in zip(coo.row, coo.col):
        vertex_u = list(graph.vertex_dict.values())[u_idx]
        vertex_v = list(graph.vertex_dict.values())[v_idx]
        u = vertex_u.id
        v = vertex_v.id
        if u != v:
            G_nx.add_edge(u, v)

    pos = nx.kamada_kawai_layout(G_nx)
    plt.figure(figsize=(8, 6))
    if isinstance(graph.vertex_dict[0], GraphVertex):
        labels = {vertex.id: vertex.global_vertex.id for vertex in graph.vertex_dict.values()}
        nx.draw(G_nx, pos, labels=labels, node_color=color, node_size=500, font_weight='bold')
    else:
        nx.draw(G_nx, pos, with_labels=True, node_color=color, node_size=500, font_weight='bold')
    plt.title("Graph")
    plt.show()


def generate_grpah(rows, cols, connection_prob=0.8, perturbation=0.3):
    G = nx.grid_2d_graph(rows, cols)

    edges = list(G.edges())
    random.shuffle(edges)
    
    for u, v in edges:
        if random.random() > connection_prob:
            if G.degree(u) > 1 and G.degree(v) > 1:
                G.remove_edge(u, v)
    
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
    
    pos = {}
    for i in range(rows):
        for j in range(cols):
            if (i, j) in G.nodes():
                x = j + random.uniform(-perturbation, perturbation)
                y = i + random.uniform(-perturbation, perturbation)
                pos[(i, j)] = (x, y)
    
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    
    new_pos = {mapping[node]: pos[node] for node in pos}
    
    return G, new_pos

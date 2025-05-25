from scipy.sparse import coo_matrix
from itertools import chain
from typing import Iterable, Set, Dict, List
from collections import deque
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time
import json

#generate graph with max 4 deg, made by ai
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


class Vertex:
    def __init__(self, id: int):
        self.id = id
        self.degree = 0
        self.adj = list()
    
    def __repr__(self):
        return f"Vertex {self.id}"

class Graph:
    def __init__(self, adj_matrix: np.ndarray = None):
        if adj_matrix is None:
            self.adjacency_matrix = None
            self.vertex_dict = {}
            return
        self.adjacency_matrix = coo_matrix(adj_matrix)
        self.vertex_dict: Dict[int, Vertex] = {}

        for idx in range(self.adjacency_matrix.shape[0]):
            self.vertex_dict[idx] = Vertex(idx)

        row = self.adjacency_matrix.row
        col = self.adjacency_matrix.col

        for r, c in zip(row, col):
            self.vertex_dict[r].adj.append(self.vertex_dict[c])
            self.vertex_dict[r].degree += 1
    
    def get_vertex(self, vertex_id: int) -> Vertex:
        return self.vertex_dict.get(vertex_id)
    
    def get_vertices(self, ids: Iterable[int]) -> List[Vertex]:
        return [self.vertex_dict[i] for i in ids if i in self.vertex_dict]
    
    def all_vertex_ids(self) -> Set[int]:
        return set(self.vertex_dict.keys())

def save_as_json(graph : Graph, filename: str):
    graph_data = {
        "vertices": [
            {
                "id": vertex.id,
                "degree": vertex.degree,
                "adj": [v.id for v in vertex.adj]
            }
            for vertex in graph.vertex_dict.values()
        ],
        "coo_matrix": {
            "row": graph.adjacency_matrix.row.tolist(),
            "col": graph.adjacency_matrix.col.tolist(),
            "data": graph.adjacency_matrix.data.tolist(),
            "shape": graph.adjacency_matrix.shape
        }
    }
    with open(filename, "w") as f:
        json.dump(graph_data, f, indent=2)

def load_from_json(filename: str) -> "Graph":
    with open(filename, "r") as f:
        data = json.load(f)
    row = np.array(data["coo_matrix"]["row"])
    col = np.array(data["coo_matrix"]["col"])
    matrix_data = np.array(data["coo_matrix"]["data"])
    shape = tuple(data["coo_matrix"]["shape"])
    adj_matrix = coo_matrix((matrix_data, (row, col)), shape=shape)
    graph = Graph()
    graph.adjacency_matrix = adj_matrix
    for vertex_data in data["vertices"]:
        vertex = Vertex(vertex_data["id"])
        graph.vertex_dict[vertex.id] = vertex

    row = adj_matrix.row
    col = adj_matrix.col

    for r, c in zip(row, col):
        graph.vertex_dict[r].adj.append(graph.vertex_dict[c])
        graph.vertex_dict[r].degree += 1

    return graph


def visualize_graph(graph, color='red'):
    G_nx = nx.Graph()

    for vertex in graph.vertex_dict.values():
        G_nx.add_node(vertex.id)

    coo = graph.adjacency_matrix.tocoo()
    for u_idx, v_idx in zip(coo.row, coo.col):
        u = list(graph.vertex_dict.values())[u_idx].id
        v = list(graph.vertex_dict.values())[v_idx].id
        if u != v:
            G_nx.add_edge(u, v)

    pos = nx.kamada_kawai_layout(G_nx)
    plt.figure(figsize=(8, 6))
    nx.draw(G_nx, pos, with_labels=True, node_color=color, node_size=500, font_weight='bold')
    plt.title("Graph")
    plt.show()

def maximal_independent_set(graph: Graph) -> Set[Vertex]:#should be log(n)
    independent_set = set()
    remaining_vertices = set(graph.vertex_dict.values())
    
    while remaining_vertices:
        subset = set()
        subset.update([vertex for vertex in remaining_vertices if random.random() < 1 / (2 * vertex.degree)])
        
        to_remove = set()
        for vertex in subset:            
            for neighbor in vertex.adj:
                if neighbor not in subset:
                    continue
                if vertex.degree > neighbor.degree:
                    to_remove.add(neighbor)
                elif vertex.degree < neighbor.degree:
                    to_remove.add(vertex)
                else:
                    if vertex.id > neighbor.id:
                        to_remove.add(neighbor)
                    else:
                        to_remove.add(vertex)
        
        subset.difference_update(to_remove)
        independent_set.update(subset)
        neighbors_to_remove = [neighbor for vertex in subset for neighbor in vertex.adj]
        remaining_vertices.difference_update(subset)
        remaining_vertices.difference_update(neighbors_to_remove)
    
    return independent_set


def construct_mis_graph(original_graph: Graph, independent_set: Set[Vertex]) -> Graph:
    adjacency_matrix = np.zeros((len(independent_set), len(independent_set)), dtype=int)
    vertex_dict = {vertex: i for i, vertex in enumerate(independent_set)}
    
    start_mis_vertex = random.choice(list(independent_set))
    mis_visited = set()
    mis_queue = deque()
    mis_queue.append(start_mis_vertex)
    
    while mis_queue:
        current_mis_vertex = mis_queue.popleft()
        mis_visited.add(current_mis_vertex.id)
        
        visited: Set[Vertex] = set()
        depth: Dict[Vertex, int] = {}
        queue = deque()

        visited.add(current_mis_vertex)
        depth[current_mis_vertex] = 0
        queue.append(current_mis_vertex)

        while queue:
            current = queue.popleft()
            current_depth = depth[current]
            
            if current_depth >= 4:
                continue
            
            for neighbor in current.adj:
                if neighbor not in visited:
                    visited.add(neighbor)
                    depth[neighbor] = current_depth + 1
                    queue.append(neighbor)
            
            if current in independent_set and current.id not in mis_visited:
                mis_queue.append(current)
                adjacency_matrix[vertex_dict[current],
                                           vertex_dict[current_mis_vertex]] = 1
                adjacency_matrix[vertex_dict[current_mis_vertex],
                                           vertex_dict[current]] = 1
    g = Graph()
    g.adjacency_matrix = coo_matrix(adjacency_matrix)
    g.vertex_dict = {vertex_dict[vertex]: vertex for vertex in vertex_dict.keys()}
    return g
    

n = 10

#start = time.perf_counter()
#G, pos = generate_connected_grid_like_planar_graph(n, n, 0.8, 0.3)
#end = time.perf_counter()
#print(f"Graph generation time: {end - start} seconds")
#g = Graph(nx.adjacency_matrix(G))

g = load_from_json("graph100vertices.json")

start = time.perf_counter()
independent_set = maximal_independent_set(g)
end = time.perf_counter()
print(f"My mis time: {end - start} seconds")

G = nx.from_scipy_sparse_array(g.adjacency_matrix)

start = time.perf_counter()
mis_networkx = nx.maximal_independent_set(G)
end = time.perf_counter()
print(f"Networkx mis time: {end - start} seconds")

#save_as_json(g, "graph100vertices.json")

colors = ['red' if i in [x.id for x in independent_set] else 'blue' for i in range(n*n)]
visualize_graph(g, colors)

start = time.perf_counter()
mis_graph = construct_mis_graph(g, independent_set)
end = time.perf_counter()
print(f"Connecting mis time: {end - start} seconds")

#visualize_graph(mis_graph)


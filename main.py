from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import laplacian
from collections import deque
import random
import numpy as np
import networkx as nx
import json
import matplotlib.pyplot as plt


class Vertex:
    def __init__(self, id):
        self.id = id
        self.adj = []
        self.degree = 0
        self.coarse = False
    
    def __str__(self):
        return f"Vertex: {self.id}"
    
    def __repr__(self):
        return f"Vertex: {self.id}"

    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return isinstance(other, Vertex) and self.id == other.id
    
class GraphVertex(Vertex):
    def __init__(self, vertex=None, id: int = None):
        super().__init__(id)
        self.global_vertex = vertex
        self.graph = None
    
class Graph:
    def __init__(self, vertex_list = None, adj_matrix = None):
        if adj_matrix is not None:
            self.adjacency_matrix = adj_matrix
            self.vertex_dict = {}

            for idx in range(self.adjacency_matrix.shape[0]):
                self.vertex_dict[idx] = Vertex(id=idx)

            row = self.adjacency_matrix.row
            col = self.adjacency_matrix.col

            for r, c in zip(row, col):
                self.vertex_dict[r].adj.append(self.vertex_dict[c])
                self.vertex_dict[r].degree += 1
        elif vertex_list is not None:
            self.vertex_dict = {i: vertex for i, vertex in enumerate(vertex_list)}
            vertex_dict_reversed = {vertex: i for i, vertex in self.vertex_dict.items()}
            
            row = []
            col = []
            
            for vertex in self.vertex_dict.values():
                for neighbor in vertex.adj:
                    if neighbor not in vertex_dict_reversed:
                        continue
                    row.append(vertex_dict_reversed[vertex])
                    col.append(vertex_dict_reversed[neighbor])
                    
            self.adjacency_matrix = coo_matrix((np.ones(len(row)), (row, col)), shape=(len(self.vertex_dict), len(self.vertex_dict)))
        
        self.laplacian_matrix = coo_matrix(laplacian(self.adjacency_matrix))
        
    def maximal_independent_set(self):
        independent_set = set()
        remaining_vertices = set(self.vertex_dict.values())
        
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
            
        for vertex in independent_set:
            vertex.coarse = True
        
        return independent_set
    
    def sort_by_coarse_matrix(self):
        
        sorted_vertices = sorted(self.vertex_dict.values(), key=lambda x: x.coarse, reverse=True)
        
        num_coarse = sum(1 for vertex in sorted_vertices if vertex.coarse)
        num_fine = len(sorted_vertices) - num_coarse
        
        vertex_dict = {i: vertex for i, vertex in enumerate(sorted_vertices)}
        vertex_dict_reversed = {vertex: i for i, vertex in vertex_dict.items()}
        
        row = []
        col = []
        
        for vertex in vertex_dict.values():
            for neighbor in vertex.adj:
                if neighbor not in vertex_dict_reversed:
                    continue
                row.append(vertex_dict_reversed[vertex])
                col.append(vertex_dict_reversed[neighbor])
                
        adjacency_matrix = coo_matrix((np.ones(len(row)), (row, col)), shape=(len(vertex_dict), len(vertex_dict)))
        
        adjacency_matrix_laplacian = laplacian(adjacency_matrix).toarray()
        a11 = adjacency_matrix_laplacian[0:num_coarse, 0:num_coarse]
        a22 = adjacency_matrix_laplacian[num_coarse:, num_coarse:]
        a21 = adjacency_matrix_laplacian[num_coarse:, 0:num_coarse]
        a12 = adjacency_matrix_laplacian[0:num_coarse, num_coarse:]
        
        print("-" * 20)
        print(f"Number of coarse vertices: {num_coarse}")
        print(adjacency_matrix_laplacian)
        a22_inv = np.linalg.pinv(a22)
        print("-"   * 20)
        print(a22)
        print(a22_inv)
        print("-"   * 20)
        print(a11 - a12 @ a22_inv @ a21)
        
        return adjacency_matrix
    
def construct_mis_graph(independent_set):
    vertex_dict = {}
    for i, vertex in enumerate(independent_set):
        vertex_dict[vertex] = GraphVertex(vertex = vertex if isinstance(vertex, Vertex) else vertex.global_vertex , id=i)
        
    for vertex in vertex_dict.keys(): 
        
        vertex_list = []
        
        visited = set()
        depth = {}
        queue = deque()

        visited.add(vertex)
        depth[vertex] = 0
        queue.append(vertex)
        while queue:
            current = queue.popleft()
            current_depth = depth[current]
            
            if current_depth <= 2:
                vertex_list.append(current)
            
            if current in independent_set and current != vertex:
                vertex_dict[vertex].adj.append(vertex_dict[current])
                vertex_dict[vertex].degree += 1
            
            if current_depth >= 3:
                continue
            
            for neighbor in current.adj:
                if neighbor not in visited:
                    visited.add(neighbor)
                    depth[neighbor] = current_depth + 1
                    queue.append(neighbor)
            
        vertex_dict[vertex].graph = Graph(vertex_list=vertex_list)
    
    g = Graph(vertex_list=list(vertex_dict.values()))
    return g
    
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

def save_as_json(graph : Graph, filename: str):
    graph_data = {
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
    graph = Graph(adj_matrix=adj_matrix)
    return graph

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
    
n = 10

#G, pos = generate_connected_grid_like_planar_graph(n, n, 0.8, 0.3)
#g = Graph(nx.adjacency_matrix(G))

g = load_from_json("graph100vertices.json")
independent_set = g.maximal_independent_set()
#save_as_json(g, "graph100vertices.json")
colors = ['red' if i in [x.id for x in independent_set] else 'blue' for i in range(n*n)]
visualize_graph(g, colors)
mis_graph = construct_mis_graph(independent_set)
for vertex in mis_graph.vertex_dict.values():
    vertex.graph.sort_by_coarse_matrix()
visualize_graph(mis_graph)




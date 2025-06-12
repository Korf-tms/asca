from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import laplacian
from numpy import ones, zeros, linalg, array
from collections import deque
import random
import json

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
    def __init__(self):
        self.adjacency_matrix = None
        self.vertex_dict = None
        self.vertex_dict_reversed = None

    def vertex_list_to_matrix(self, vertex_list = None):
        if vertex_list is None:
            vertex_dict = self.vertex_dict_reversed
        else:
            vertex_dict = {vertex:i for i, vertex in enumerate(vertex_list)}
            
        row = []
        col = []
        
        for vertex in vertex_dict.keys():
            for neighbor in vertex.adj:
                if neighbor not in vertex_dict:
                    continue
                row.append(vertex_dict[vertex])
                col.append(vertex_dict[neighbor])
                
        return coo_matrix((ones(len(row)), (row, col)), shape=(len(vertex_dict), len(vertex_dict)))

    
class OriginalGraph(Graph):
    def __init__(self, filename, adjacency_matrix = None):
        if adjacency_matrix is not None:
            self.adjacency_matrix = coo_matrix(adjacency_matrix)
            return
        with open(filename, "r") as f:
            data = json.load(f)
            row = array(data["coo_matrix"]["row"])
            col = array(data["coo_matrix"]["col"])
            shape = tuple(data["coo_matrix"]["shape"])
        
        matrix_data = ones(len(col))
        self.adjacency_matrix = coo_matrix((matrix_data, (row, col)), shape=shape)
        self.vertex_dict = {}
        self.vertex_dict_reversed = {}

        for idx in range(self.adjacency_matrix.shape[0]):
            self.vertex_dict[idx] = Vertex(id=idx)
            self.vertex_dict_reversed[self.vertex_dict[idx]] = idx

        row = self.adjacency_matrix.row
        col = self.adjacency_matrix.col

        for r, c in zip(row, col):
            self.vertex_dict[r].adj.append(self.vertex_dict[c])
            self.vertex_dict[r].degree += 1
        
    def maximal_independent_set(self):
        #returns set of coarse vertices
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
    
    def save_as_json(self):
        graph_data = {
            "coo_matrix": {
                "row": self.adjacency_matrix.row.tolist(),
                "col": self.adjacency_matrix.col.tolist(),
                "shape": self.adjacency_matrix.shape
            }
        }
        with open(f"graph{self.adjacency_matrix.shape[0]}x{self.adjacency_matrix.shape[1]}.json", "w") as f:
            json.dump(graph_data, f, indent=2)
    
    

class CoarseGraph(Graph):
    def __init__(self, independent_set, graph):
        #constructs new graph from the independent set, connects all coarse vertices that are max 3 hops apart
        #each vertex V in this graph has its own subgraph
        #subgraph contains vertices that are max 2 depth from V
        original_vertex_to_vertex = {}
        for iterator, vertex in enumerate(independent_set):
            original_vertex_to_vertex[vertex] = GraphVertex(vertex = vertex, id=iterator)#creates new vertex with global vertex from the original graph

        for vertex in original_vertex_to_vertex.keys():
            vertex_list = []#subgraph

            visited = set()
            depth = {}
            queue = deque()

            visited.add(vertex)
            depth[vertex] = 0
            queue.append(vertex)
            while queue:#bfs
                current = queue.popleft()
                current_depth = depth[current]

                if current_depth <= 2:
                    vertex_list.append(current)#adding vertex to the subgraph

                if current in independent_set and current != vertex:#connecting the coarse vertices
                    original_vertex_to_vertex[vertex].adj.append(original_vertex_to_vertex[current])
                    original_vertex_to_vertex[vertex].degree += 1

                if current_depth >= 3:
                    continue
                
                for neighbor in current.adj:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        depth[neighbor] = current_depth + 1
                        queue.append(neighbor)

            original_vertex_to_vertex[vertex].graph = SubGraph(vertex_list=vertex_list, graph=graph)
        self.vertex_dict = {}
        self.vertex_dict_reversed = {}
        for iterator, vertex in enumerate(original_vertex_to_vertex.values()):
            self.vertex_dict[iterator] = vertex
            self.vertex_dict_reversed[vertex] = iterator
        self.adjacency_matrix = self.vertex_list_to_matrix(list(original_vertex_to_vertex.values()))
        self.global_graph = graph

class SubGraph(Graph):
    def __init__(self, vertex_list, graph):
        self.vertex_dict = {i: vertex for i, vertex in enumerate(vertex_list)}
        self.vertex_dict_reversed = {vertex: i for i, vertex in enumerate(vertex_list)}
        self.num_coarse = sum(1 for vertex in self.vertex_dict.values() if vertex.coarse)
        self.adjacency_matrix = self.vertex_list_to_matrix()
        self.sorted_vertices = sorted(self.vertex_dict.values(), key=lambda x: x.coarse, reverse=True)
        self.global_graph = graph
    
    def local_schur_complement(self):
        adjacency_matrix = self.vertex_list_to_matrix(self.sorted_vertices)

        adjacency_matrix_laplacian = laplacian(adjacency_matrix).toarray()
        a11 = adjacency_matrix_laplacian[0:self.num_coarse, 0:self.num_coarse]
        a22 = adjacency_matrix_laplacian[self.num_coarse:, self.num_coarse:]
        a21 = adjacency_matrix_laplacian[self.num_coarse:, 0:self.num_coarse]
        a12 = adjacency_matrix_laplacian[0:self.num_coarse, self.num_coarse:]
        a22_inv = linalg.pinv(a22)
        
        return a11 - a12 @ a22_inv @ a21
    
    def local_to_global_mapping(self):
        #row is true global vertex
        #column is local coarse
        local_to_global_mapping_matrix = zeros((len(self.global_graph.vertex_dict.values()), self.num_coarse))
        coarse = [x for x in self.sorted_vertices if x.coarse]
        
        for i, vertex in enumerate(coarse):
            local_to_global_mapping_matrix[self.global_graph.vertex_dict_reversed[vertex]][i] = 1
        
        return coo_matrix(local_to_global_mapping_matrix)
    
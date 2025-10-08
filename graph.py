from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import laplacian
from numpy import ones, zeros, linalg, array
from collections import deque
import random
import csv
import logging
import pandas as pd 
import numpy as np

logger = logging.getLogger(__name__)

class Vertex:
    def __init__(self, id):
        self.id = id
        self.adj = []
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
    def __init__(self, id: int = None, vertex: Vertex = None):
        super().__init__(id)
        self.graph = None
        self.original_vertex = vertex

class SubgraphVertex(Vertex):
    def __init__(self, id: int = None, vertex : Vertex = None):
        super().__init__(id)
        self.original_vertex = vertex
    


#-------------------------------------------------------------------------------------------------------------------
class Graph:
    def __init__(self, vertex_list = []):
        self.vertex_list = vertex_list
        self.name = "MainGraph"

    def init_from_csv(self, path):
        df = pd.read_csv(path)
        rows = df['row'].to_numpy()
        cols = df['col'].to_numpy()
        n = int(max(rows.max(), cols.max()) + 1)
        vertex_dictionary = {i: Vertex(i) for i in range(n)}
        for r, c in zip(rows, cols):
            vr = vertex_dictionary[int(r)]
            vc = vertex_dictionary[int(c)]
            vr.adj.append(vc)
            vc.adj.append(vr)

        self.vertex_list = list(vertex_dictionary.values())

    def vertex_list_to_adj_matrix(self, vertex_list):
        row = []
        col = []
        val = []
        for vertex in vertex_list:
            for neighbor in vertex.adj:
                row.append(vertex.id)
                col.append(neighbor.id)
                val.append(1)

        shape = len(vertex_list)
        temp = np.zeros((shape, shape), dtype=float)
        temp[row, col] = val
        return temp

    def maximal_independent_set(self):
        independent_set = set()
        remaining_vertices = set(self.vertex_list)
        
        while remaining_vertices:
            subset = set()
            subset.update([vertex for vertex in remaining_vertices if random.random() < 1 / (2 * len(vertex.adj))])
            
            to_remove = set()
            for vertex in subset:            
                for neighbor in vertex.adj:
                    if neighbor not in subset:
                        continue
                    if len(vertex.adj) > len(neighbor.adj):
                        to_remove.add(neighbor)
                    elif len(vertex.adj) < len(neighbor.adj):
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

    def local_schur_complement(self):
        num_coarse = sum(1 for vertex in self.vertex_list if vertex.coarse)
        sorted_vertices = sorted(self.vertex_list, key=lambda x: x.coarse, reverse=True)

        adjacency_matrix = self.vertex_list_to_adj_matrix(sorted_vertices)

        adjacency_matrix_laplacian = laplacian(adjacency_matrix)

        a11 = adjacency_matrix_laplacian[0:num_coarse, 0:num_coarse]
        a22 = adjacency_matrix_laplacian[num_coarse:, num_coarse:]
        a21 = adjacency_matrix_laplacian[num_coarse:, 0:num_coarse]
        a12 = adjacency_matrix_laplacian[0:num_coarse, num_coarse:]

        return a11 - a12 @ linalg.solve(a22, a21)

class CoarseGraph(Graph):
    def __init__(self, independent_set, graph : Graph):
        self.name = "CoarseGraph"
        logging.info("Creating coarse graph")
        original_vertex_to_graph_vertex = {}
        temp = len(graph.vertex_list)
        self.count_matrix = np.zeros((temp, temp), dtype=float)
        index = 0
        for iterator, original_vertex in enumerate(independent_set):
            original_vertex_to_graph_vertex[original_vertex] = GraphVertex(id=iterator, vertex=original_vertex)

        for original_vertex in original_vertex_to_graph_vertex.keys():
            vertex_list = [] 

            visited = set()
            depth = {}
            queue = deque()

            visited.add(original_vertex)
            depth[original_vertex] = 0
            queue.append(original_vertex)
            while queue:
                current = queue.popleft()
                current_depth = depth[current]

                if current in independent_set and current != original_vertex and current_depth >= 3:#connecting the coarse vertices
                    original_vertex_to_graph_vertex[original_vertex].adj.append(original_vertex_to_graph_vertex[current])

                if current_depth >= 4:
                    continue

                vertex_list.append(current)#creating the subgraph

                for neighbor in current.adj:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        depth[neighbor] = current_depth + 1
                        queue.append(neighbor)

            original_vertex_to_graph_vertex[original_vertex].graph = SubGraph(vertex_list=vertex_list, graph=self, name=f"SubGraph{index}")
            index += 1

        self.vertex_list = list(original_vertex_to_graph_vertex.values())
        self.vertex_list = sorted(self.vertex_list, key=lambda x: x.original_vertex.coarse, reverse=True)
        self.coarse_indexed_list = {vertex: i for i, vertex in enumerate(original_vertex_to_graph_vertex.keys())}
        self.parent = graph
        logging.info(f"coarse graph: {self.vertex_list}")
        for vertex in self.vertex_list:
            logging.info(f"{vertex} adj: {vertex.adj}")
        logging.info(f"count matrix:\n{self.count_matrix}")

class SubGraph(Graph):
    def __init__(self, vertex_list, graph, name):
        logging.info("creating subgraph")
        original_vertex_to_subgraph_vertex = {original_vertex: SubgraphVertex(id= i, vertex=original_vertex) 
                                              for i, original_vertex 
                                              in enumerate(sorted(vertex_list, key=lambda x: x.coarse, reverse=True))}

        #connecting the local subGraphVertex graph
        for vertex in vertex_list:
            for neighbour in vertex.adj:
                if neighbour not in vertex_list:
                    continue
                original_vertex_to_subgraph_vertex[vertex].adj.append(original_vertex_to_subgraph_vertex[neighbour])
                graph.count_matrix[vertex.id, neighbour.id] += 1
        
        self.name = name
        self.vertex_list =  list(original_vertex_to_subgraph_vertex.values())
        self.num_coarse = sum(1 for vertex in self.vertex_list if vertex.original_vertex.coarse)
        self.parent : CoarseGraph = graph
        logging.info(f"vertex list: {self.vertex_list}, num coarse: {self.num_coarse}")
    
    def local_schur_complement(self):

        logging.info("computing local schur complement")
        adjacency_matrix = self.vertex_list_to_adj_matrix(self.vertex_list)
        logging.info(f"adj matrix before:\n {adjacency_matrix}")

        test = set()
        for vertex in self.vertex_list:
            for neighbour in vertex.adj:
                if((vertex.id, neighbour.id) in test):
                    continue
                test.add((vertex.id, neighbour.id))
                adjacency_matrix[vertex.id, neighbour.id] = adjacency_matrix[vertex.id, neighbour.id] / self.parent.count_matrix[vertex.original_vertex.id, neighbour.original_vertex.id]

        logging.info(f"adj matrix after:\n {adjacency_matrix}")
        adjacency_matrix_laplacian = laplacian(adjacency_matrix)
        logging.info(f"adj matrix laplacian:\n {adjacency_matrix_laplacian}")
        a11 = adjacency_matrix_laplacian[0:self.num_coarse, 0:self.num_coarse]
        a22 = adjacency_matrix_laplacian[self.num_coarse:, self.num_coarse:]
        a21 = adjacency_matrix_laplacian[self.num_coarse:, 0:self.num_coarse]
        a12 = adjacency_matrix_laplacian[0:self.num_coarse, self.num_coarse:]
        temp = a11 - a12 @ linalg.solve(a22, a21)
        logging.info(f"schur complement:\n{temp}")
        return temp
    
    def local_to_global_mapping(self):
        #row is true global vertex
        #column is local coarse
        local_to_global_mapping_matrix = zeros((len(self.parent.vertex_list), self.num_coarse))
        coarse = [x for x in self.vertex_list if x.original_vertex.coarse]
        for vertex in coarse:
            local_to_global_mapping_matrix[self.parent.coarse_indexed_list[vertex.original_vertex]][vertex.id] = 1
        return coo_matrix(local_to_global_mapping_matrix)
    
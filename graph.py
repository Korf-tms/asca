from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import laplacian
from numpy import zeros, linalg
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
        self.name = ""
    
    def __str__(self):
        return f"{self.name}Vertex: {self.id}"
    
    def __repr__(self):
        return f"{self.name}Vertex: {self.id}"

    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return isinstance(other, Vertex) and self.id == other.id
    
class GraphVertex(Vertex):
    def __init__(self, id: int = None, vertex: Vertex = None):
        super().__init__(id)
        self.graph = None
        self.original_vertex = vertex
        self.name = "Graph"

class SubgraphVertex(Vertex):
    def __init__(self, id: int = None, vertex : Vertex = None):
        super().__init__(id)
        self.original_vertex = vertex
        self.name = "Subgraph"
    
#-------------------------------------------------------------------------------------------------------------------
class Graph:
    def __init__(self, vertex_list = None, path = ""):
        if vertex_list != None:
            self.vertex_list = vertex_list
        elif path != "":
            split_path = path.split(".")
            if split_path[1] == "csv":
                self.init_from_csv(path)

        self.name = "MainGraph"
        self.coarse_vertices = self.select_with_spacing(1)
        for vertex in self.coarse_vertices:
            vertex.coarse = True
        self.sorted_vertex_adj_matrix_mapping = {vertex: i for i, vertex in enumerate(sorted(self.vertex_list, key=lambda x: (not x.coarse, x.id), reverse=False))}

    def init_from_csv(self, path, csv_type = 0):
        df = pd.read_csv(path)
        if csv_type == 0:
            self.vertex_list = self.__init_from_csv_coo(df)
        #if csv_type == 1:
        #   non compressed matrix

    def __init_from_csv_coo(self, dataframe):
        rows = dataframe['row'].to_numpy()
        cols = dataframe['col'].to_numpy()
        n = int(max(rows.max(), cols.max()) + 1)
        vertex_dictionary = {i: Vertex(i) for i in range(n)}
        for r, c in zip(rows, cols):
            vr = vertex_dictionary[int(r)]
            vc = vertex_dictionary[int(c)]
            vr.adj.append(vc)
            vc.adj.append(vr)

        return list(vertex_dictionary.values())

    def vertex_list_to_adj_matrix(self, vertex_list):
        row = []
        col = []
        val = []
        mapping = {vertex: i for i, vertex in enumerate(vertex_list)}
        for vertex in vertex_list:
            for neighbor in vertex.adj:
                row.append(mapping[vertex])
                col.append(mapping[neighbor])
                val.append(1)

        shape = len(vertex_list)
        temp = np.zeros((shape, shape), dtype=float)
        temp[row, col] = val
        return temp

    def select_with_spacing(self, spacing = 1):
        coarse_vertices = set()
        visited = set()
        vertex_matrix = np.array(self.vertex_list)
        vertex_matrix = vertex_matrix.reshape((5, 5))
        for row, row_array in enumerate(vertex_matrix):
            for col, vertex in enumerate(row_array):
                if vertex in visited:
                    continue
                if vertex not in coarse_vertices:
                    coarse_vertices.add(vertex)
                    row_start = max(0, row - spacing)
                    row_end   = min(vertex_matrix.shape[0], row + spacing + 1)
                    column_start = max(0, col - spacing)
                    column_end = min(vertex_matrix.shape[1], col + spacing + 1)
                    visited.update(vertex_matrix[row_start:row_end, column_start:column_end].ravel())
        return coarse_vertices

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
        return independent_set

    def local_schur_complement(self):
        num_coarse = len(self.coarse_vertices)
        if num_coarse == 0:
            return
        sorted_vertices = sorted(self.vertex_list, key=lambda x: (not x.coarse, x.id), reverse=False)
        adjacency_matrix = self.vertex_list_to_adj_matrix(sorted_vertices)
        adjacency_matrix_laplacian = laplacian(adjacency_matrix)
        a11 = adjacency_matrix_laplacian[0:num_coarse, 0:num_coarse]
        a22 = adjacency_matrix_laplacian[num_coarse:, num_coarse:]
        a21 = adjacency_matrix_laplacian[num_coarse:, 0:num_coarse]
        a12 = adjacency_matrix_laplacian[0:num_coarse, num_coarse:]

        return a11 - a12 @ linalg.solve(a22, a21)

class CoarseGraph(Graph):
    def __init__(self, coarse_vertices, parent_graph : Graph):
        #should create new class for grid graphs
        #with vertex_matrix and coord dict so we dont have to search
        #also create some application class to to handle the different inputs instead of Graph
        self.name = "CoarseGraph"
        self.count_matrix = dict()
        self.parent : Graph = parent_graph
        self.vertex_list = [GraphVertex(id=iterator, vertex=original_vertex) for iterator, original_vertex in enumerate(coarse_vertices)]
        self.create_subgraphs_grid_all()

    def connect_coarse_graph_by_depth(self, max_depth = 3):
        original_to_local_vertex = {vertex.original_vertex: vertex for vertex in self.vertex_list}
        for vertex in self.vertex_list:
            visited = set()
            depth = {}
            queue = deque()

            visited.add(vertex.original_vertex)
            depth[vertex.original_vertex] = 0
            queue.append(vertex.original_vertex)

            while queue:
                current = queue.popleft()
                current_depth = depth[current]

                if current_depth > max_depth:
                    continue
                if current.coarse and current != vertex.original_vertex:#connecting the coarse vertices
                    vertex.adj.append(original_to_local_vertex[current])

                for neighbor in current.adj:
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    depth[neighbor] = current_depth + 1
                    queue.append(neighbor)

    def create_subgraphs_by_depth(self, max_depth = 3):
        for iterator, vertex in enumerate(self.vertex_list):
            subgraph_vertex_list = [] 

            visited = set()
            depth = {}
            queue = deque()

            visited.add(vertex.original_vertex)
            depth[vertex.original_vertex] = 0
            queue.append(vertex.original_vertex)
            while queue:
                current = queue.popleft()
                current_depth = depth[current]

                if current_depth > max_depth:
                    continue

                subgraph_vertex_list.append(current)

                for neighbor in current.adj:
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    depth[neighbor] = current_depth + 1
                    queue.append(neighbor)

            vertex.graph = SubGraph(vertex_list=subgraph_vertex_list, graph=self, name=f"SubGraph{iterator}")

    def create_subgraphs_grid_around_coarse(self, size = 1):
        vertex_matrix = np.array(self.parent.vertex_list)
        vertex_matrix = vertex_matrix.reshape((5, 5))
        for iterator, vertex in enumerate(self.vertex_list):
            res = np.where(vertex_matrix == vertex.original_vertex)
            row = res[0][0]
            col = res[1][0]
            row_start = max(0, row - size)
            row_end   = min(vertex_matrix.shape[0], row + size + 1)
            column_start = max(0, col - size)
            column_end = min(vertex_matrix.shape[1], col + size + 1)
            subgraph_vertex_list = vertex_matrix[row_start:row_end, column_start:column_end].ravel()
            vertex.graph = SubGraph(vertex_list=subgraph_vertex_list, graph=self, name=f"SubGraph{iterator}")

    def create_subgraphs_grid_all(self, size = 1):
        vertex_matrix = np.array(self.parent.vertex_list)
        vertex_matrix = vertex_matrix.reshape((5, 5))
        vertex_matrix_center = vertex_matrix[size:-size, size:-size]
        self.vertex_list = [GraphVertex(id=iterator, vertex=original_vertex) for iterator, original_vertex in enumerate(vertex_matrix_center.ravel())]#this is not optimal need to change
        #before all subgraphs were build around coarse vertice, in this case we need to use non coarse vertices
        for iterator, vertex in enumerate(self.vertex_list):
            res = np.where(vertex_matrix == vertex.original_vertex)
            row = res[0][0]
            col = res[1][0]
            row_start = max(0, row - size)
            row_end   = min(vertex_matrix.shape[0], row + size + 1)
            column_start = max(0, col - size)
            column_end = min(vertex_matrix.shape[1], col + size + 1)
            subgraph_vertex_list = vertex_matrix[row_start:row_end, column_start:column_end].ravel()
            vertex.graph = SubGraph(vertex_list=subgraph_vertex_list, graph=self, name=f"SubGraph{iterator}")

class SubGraph(Graph):
    def __init__(self, vertex_list, graph, name):
        self.name = name
        self.vertex_list =  list()
        self.parent : CoarseGraph = graph
        
        original_vertex_to_subgraph_vertex = dict()
        for i, original_vertex in enumerate(vertex_list):
            subgraph_vertex = SubgraphVertex(id= i, vertex=original_vertex) 
            original_vertex_to_subgraph_vertex[original_vertex] = subgraph_vertex
            self.vertex_list.append(subgraph_vertex)

        for vertex in self.vertex_list:
            vertex.adj.extend([original_vertex_to_subgraph_vertex[original_vertex] for original_vertex in list(set(vertex.original_vertex.adj) & set(vertex_list))])

        for vertex in self.vertex_list:
            for neighbour in vertex.adj:
                graph.count_matrix[(vertex.original_vertex.id, neighbour.original_vertex.id)] = graph.count_matrix.get((vertex.original_vertex.id, neighbour.original_vertex.id), 0) + 1

        self.num_coarse = sum(1 for vertex in self.vertex_list if vertex.original_vertex.coarse)
        self.sorted_vertex_list = sorted(self.vertex_list, key=lambda x: (not x.original_vertex.coarse, x.id), reverse=False)
    

    def local_schur_complement(self):
        adjacency_matrix = self.vertex_list_to_adj_matrix(self.sorted_vertex_list)

        vertex_ajd_matrix_mapping = {vertex: i for i, vertex in enumerate(self.sorted_vertex_list)}
        for vertex in self.vertex_list:
            for neighbour in vertex.adj:
                adjacency_matrix[vertex_ajd_matrix_mapping[vertex], 
                                 vertex_ajd_matrix_mapping[neighbour]] /= self.parent.count_matrix[(vertex.original_vertex.id, 
                                                                                                    neighbour.original_vertex.id)]
        adjacency_matrix_laplacian = laplacian(adjacency_matrix)
        a11 = adjacency_matrix_laplacian[0:self.num_coarse, 0:self.num_coarse]
        a22 = adjacency_matrix_laplacian[self.num_coarse:, self.num_coarse:]
        a21 = adjacency_matrix_laplacian[self.num_coarse:, 0:self.num_coarse]
        a12 = adjacency_matrix_laplacian[0:self.num_coarse, self.num_coarse:]
        temp = a11 - a12 @ linalg.solve(a22, a21)
        return temp
    
    def local_to_global_mapping(self):
        #row is true global vertex
        #column is local coarse
        local_to_global_mapping_matrix = zeros((len(self.parent.vertex_list), self.num_coarse))
        coarse = sorted([x for x in self.vertex_list if x.original_vertex.coarse], key=lambda x: x.id)
        for i, vertex in enumerate(coarse):
            local_to_global_mapping_matrix[self.parent.parent.sorted_vertex_adj_matrix_mapping[vertex.original_vertex]][i] = 1
        return coo_matrix(local_to_global_mapping_matrix)
    
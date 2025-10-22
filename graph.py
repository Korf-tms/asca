from scipy.sparse.csgraph import laplacian
from collections import deque
from scipy.sparse import coo_matrix

import pathlib as pl
import pandas as pd
import numpy as np

import random

class Vertex:
    def __init__(self, id):
        self.id = id
        self.adj = []
        self.coarse = False
        self.name = ""
        self.graph = None
    
    def __str__(self):
        return f"{self.name}Vertex: {self.id}"
    
    def __repr__(self):
        return f"{self.name}Vertex: {self.id}"

    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return isinstance(other, Vertex) and self.id == other.id

class SubgraphVertex(Vertex):
    def __init__(self, id: int, vertex : Vertex):
        super().__init__(id)
        self.original_vertex = vertex
        self.name = "Subgraph"

class Graph:
    """
    Base graph class
    vertex_list : list of Vertex obejects - main representation of the graph
    edge_count : dict - counts how many subgraphs each edge is part of
    coarse_vertices : list - list of coarse vertices in the graph, is populated after one of the select_coarse methods is called
    name : str - name of the graph for visualization purposes
    """
    def __init__(self, vertex_list : list[Vertex]):
        self.vertex_list = vertex_list
        self.edge_count = dict()
        self.coarse_vertices = list()
        self.name = "Graph"
    

    """
    Creates vertex list from rows, cols, values (coo_matrix format).
    """
    @staticmethod
    def __vertex_list_from_coo(rows, cols, values):

        if len(rows) != len(cols) != len(values):
            raise ValueError("Invalid COO representation")

        n = int(max(rows.max(), cols.max()) + 1)#get highest vertex index
        vertex_dictionary = {i: Vertex(i) for i in range(n)}
        for row, col in zip(rows, cols):
            vr = vertex_dictionary[int(row)]
            vc = vertex_dictionary[int(col)]
            if vc not in vr.adj:
                vr.adj.append(vc)
            if vr not in vc.adj:
                vc.adj.append(vr)
        return list(vertex_dictionary.values())

    """
    Gets coo format from csv file in format row,col,val and creates graph from it.
    """
    @classmethod
    def from_csv(cls, path : str):

        if not pl.Path(path).exists():
            raise FileNotFoundError(f"File {path} does not exist.")
        if pl.Path(path).suffix != ".csv":
            raise ValueError(f"File {path} is not a CSV file.")
        
        dataframe = pd.read_csv(path)
        rows = dataframe['row'].to_numpy()
        cols = dataframe['col'].to_numpy()
        values = dataframe['val'].to_numpy()
        return cls(cls.__vertex_list_from_coo(rows, cols, values))

    """
    Reads hdf5 file that represents graph by either coo format djacency matrix or full adjacency matrix and creates graphfrom it.
    """
    @classmethod
    def from_hdf5(cls, path : str):

        if not pl.Path(path).exists():
            raise FileNotFoundError(f"File {path} does not exist.")
        if pl.Path(path).suffix != ".hdf5":
            raise ValueError(f"File {path} is not a HDF5 file.")
        
        with pd.HDFStore(path, mode="r") as store:
            keys = set(store.keys())
            if "/coo_matrix" in keys:
                dataframe = store.get("coo_matrix")
                rows = dataframe['row'].to_numpy()
                cols = dataframe['col'].to_numpy()
                values = dataframe['val'].to_numpy()
                return cls(cls.__vertex_list_from_coo(rows, cols, values))
            elif "/adj_matrix" in keys:
                dataframe = store.get("adj_matrix")
                adj_matrix = coo_matrix(dataframe.to_numpy())
                return cls(cls.__vertex_list_from_coo(adj_matrix.row, adj_matrix.col, adj_matrix.data))
            else:
                raise ValueError(f"HDF5 file {path} does not contain 'coo_matrix' or 'adj_matrix' key.")                   

    """
    Creates adjacency matrix from given vertex list, the order of vertex list matters.
    """
    def vertex_list_to_adj_matrix(self, vertex_list):

        if len(vertex_list) == 0:
            return 0

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


    """
    computes the local schur complement for the graph
    """
    def local_schur_complement(self):
        num_coarse = len(self.coarse_vertices)
        if num_coarse == 0:
            print("Warning: no coarse vertices in graph.")
            return 0
        sorted_vertices = sorted(self.vertex_list, key=lambda x: (not x.coarse, x.id), reverse=False)
        adjacency_matrix = self.vertex_list_to_adj_matrix(sorted_vertices)
        adjacency_matrix_laplacian = laplacian(adjacency_matrix)
        a11 = adjacency_matrix_laplacian[0:num_coarse, 0:num_coarse]
        a22 = adjacency_matrix_laplacian[num_coarse:, num_coarse:]
        a21 = adjacency_matrix_laplacian[num_coarse:, 0:num_coarse]
        a12 = adjacency_matrix_laplacian[0:num_coarse, num_coarse:]

        return a11 - a12 @ np.linalg.solve(a22, a21)
    
    """
    Sets coarse vertices and creates mapping, that is used in schur complement by the subgraphs
    """
    def set_coarse(self, coarse_vertices):
        for vertex in coarse_vertices:
            vertex.coarse = True
        self.coarse_vertices = coarse_vertices
        self.sorted_vertex_adj_matrix_mapping = {vertex: i for i, vertex in enumerate(sorted(self.vertex_list, key=lambda x: (not x.coarse, x.id), reverse=False))}
    
    """
    returns list of subgraphs in the graph
    """
    def get_subgraphs(self):
        return [vertex.graph for vertex in self.vertex_list if vertex.graph != None]

class UniversalGraph(Graph):
    """
    Generic graph
    """
    def __init__(self, vertex_list):
        super().__init__(vertex_list)
    

    """
    Selects coarse vertices that are part of maximal independent set.
    Maximal independent set is a set of vertices such that no two vertices are adjacent.
    And no additional vertices can be added to the set without violating this property.
    @return coarse_vertices - set of coarse vertices
    """
    def select_coarse_mis(self):
        coarse_vertices = set()
        remaining_vertices = set(self.vertex_list)
        
        while remaining_vertices:
            current = remaining_vertices.pop()
            coarse_vertices.add(current)
            remaining_vertices.difference_update(current.adj)
        self.set_coarse(coarse_vertices)
        return coarse_vertices

    """
    Creates subgraphs around each coarse vertex with given depth.
    """
    def create_subgraphs_depth(self, max_depth = 2):
        if depth < 1:
            raise ValueError("Max depth must be at least 1.")

        for iterator, vertex in enumerate(self.coarse_vertices):
            subgraph_vertex_list = [] 

            visited = set()
            depth = {}
            queue = deque()

            visited.add(vertex)
            depth[vertex] = 0
            queue.append(vertex)

            while queue:
                current = queue.popleft()
                subgraph_vertex_list.append(current)

                for neighbor in current.adj:
                    if neighbor in visited:
                        continue
                    if depth[current] + 1 > max_depth + 1:
                        continue
                    visited.add(neighbor)
                    depth[neighbor] = depth[current] + 1
                    queue.append(neighbor)

            vertex.graph = SubGraph(vertex_list=subgraph_vertex_list, graph=self, name=f"SubGraph{iterator}")

class GridGraph(Graph):
    """
    Grid graph
    shape : tuple - shape of the graph grid (rows, cols)
    vertex_matrix : np.ndarray - matrix of vertices as they are in the grid, neighbours in the matrix means neighbours in the graph
    vertex_matrix_coords : dict - mapping of vertices to their coordinates in the vertex_matrix for easy access
    """
    def __init__(self, vertex_list, shape : tuple[int, int]):
        super().__init__(vertex_list)
        self.shape = shape
        self.vertex_matrix = np.array(vertex_list).reshape(shape)
        self.vertex_matrix_coords = dict()
        for row, col in np.ndindex(shape):
            self.vertex_matrix_coords[self.vertex_matrix[row, col]] = (row, col)

    @classmethod
    def from_csv(cls, path, shape):
        base = Graph.from_csv(path)
        return cls(base.vertex_list, shape)

    @classmethod
    def from_hdf5(cls, path, shape):
        base = Graph.from_hdf5(path)
        return cls(base.vertex_list, shape)

    """
    
    """
    def select_coarse_spacing(self, spacing = 1):
        coarse_vertices = set()
        visited = set()
        for vertex in self.vertex_list:
            if vertex in visited:
                continue
            if vertex not in coarse_vertices:
                coarse_vertices.add(vertex)
                visited.update(self.get_vertices_around(vertex, spacing))

        self.set_coarse(coarse_vertices)
        return coarse_vertices
    
    """
    """
    def select_coarse_star(self, size = 1):
        coarse_vertices = set()
        visited = set()
        for vertex in self.vertex_list:
            if vertex in visited:
                continue
            if vertex not in coarse_vertices:
                coarse_vertices.add(vertex)
                adj = set([vertex])
                for _ in range(size):
                    adj.update(*(v.adj for v in adj))
                visited.update(adj)
        self.set_coarse(coarse_vertices)
        return coarse_vertices

    """
    Selects every n-th vertice as they were inputed in vertex list.
    """
    def select_coarse_every_nth(self, n = 2):
        coarse_vertices = set()
        for iterator in range(0, len(self.vertex_list), n):
            coarse_vertices.add(self.vertex_list[iterator])
        self.set_coarse(coarse_vertices)
        return coarse_vertices

    """
    Creates subgraphs around each coarse vertex with given size, includes the corner vertices not just adjacent ones.
    """
    def create_subgraphs_around_coarse(self, size = 1):
        for iterator, vertex in enumerate(self.coarse_vertices):

            subgraph_vertex_list = self.get_vertices_around(vertex, size)

            vertex.graph = SubGraph(vertex_list=subgraph_vertex_list, graph=self, name=f"SubGraph{iterator}")

    """
    Creates the maximum possible number of subgraphs of given size,
    size = 1 means one vertice in each direction.
    """
    def create_subgraphs_max(self, size = 1):
        vertex_matrix_center = self.vertex_matrix[size:-size, size:-size]
        for iterator, vertex in enumerate(vertex_matrix_center.ravel()):

            subgraph_vertex_list = self.get_vertices_around(vertex, size)

            vertex.graph = SubGraph(vertex_list=subgraph_vertex_list, graph=self, name=f"SubGraph{iterator}")

    """
    Helper method that returns vertices around given vertex in square of given size.
    """
    def get_vertices_around(self, vertex, size = 1):
        row = self.vertex_matrix_coords[vertex][0]
        col = self.vertex_matrix_coords[vertex][1]

        row_start = max(0, row - size)
        row_end   = min(self.shape[0], row + size + 1)
        column_start = max(0, col - size)
        column_end = min(self.shape[1], col + size + 1)
        return self.vertex_matrix[row_start:row_end, column_start:column_end].ravel()

class SubGraph():
    """
    
    """
    def __init__(self, vertex_list, graph, name):
        self.name = name
        self.vertex_list =  list()
        self.parent = graph
        
        original_vertex_to_subgraph_vertex = dict()
        for i, original_vertex in enumerate(vertex_list):
            subgraph_vertex = SubgraphVertex(id= i, vertex=original_vertex) 
            original_vertex_to_subgraph_vertex[original_vertex] = subgraph_vertex
            self.vertex_list.append(subgraph_vertex)

        vertex_list_set = set(vertex_list)

        for vertex in self.vertex_list:
            vertex.adj.extend([original_vertex_to_subgraph_vertex[original_vertex] for original_vertex in vertex.original_vertex.adj if original_vertex in vertex_list_set])

        
        for vertex in self.vertex_list:
            for neighbour in vertex.adj:
                key = (vertex.original_vertex.id, neighbour.original_vertex.id)
                graph.edge_count[key] = graph.edge_count.get(key, 0) + 1

        self.num_coarse = sum(1 for vertex in self.vertex_list if vertex.original_vertex.coarse)
        self.sorted_vertex_list = sorted(self.vertex_list, key=lambda vertex: (not vertex.original_vertex.coarse, vertex.id), reverse=False)
    

    def local_schur_complement(self):
        adjacency_matrix = self.vertex_list_to_adj_matrix(self.sorted_vertex_list)

        vertex_ajd_matrix_mapping = {vertex: iterator for iterator, vertex in enumerate(self.sorted_vertex_list)}
        for vertex in self.vertex_list:
            for neighbour in vertex.adj:
                adjacency_matrix[vertex_ajd_matrix_mapping[vertex], 
                                 vertex_ajd_matrix_mapping[neighbour]] /= self.parent.edge_count[(vertex.original_vertex.id, 
                                                                                                    neighbour.original_vertex.id)]
        adjacency_matrix_laplacian = laplacian(adjacency_matrix)
        a11 = adjacency_matrix_laplacian[0:self.num_coarse, 0:self.num_coarse]
        a22 = adjacency_matrix_laplacian[self.num_coarse:, self.num_coarse:]
        a21 = adjacency_matrix_laplacian[self.num_coarse:, 0:self.num_coarse]
        a12 = adjacency_matrix_laplacian[0:self.num_coarse, self.num_coarse:]
        temp = a11 - a12 @ np.linalg.solve(a22, a21)
        return temp
    
    def local_to_global_mapping(self):
        #row is true global vertex
        #column is local coarse
        local_to_global_mapping_matrix = np.zeros((len(self.parent.coarse_vertices), self.num_coarse))
        coarse = sorted([x for x in self.vertex_list if x.original_vertex.coarse], key=lambda vertex: vertex.id)
        for i, vertex in enumerate(coarse):
            local_to_global_mapping_matrix[self.parent.sorted_vertex_adj_matrix_mapping[vertex.original_vertex]][i] = 1
        return coo_matrix(local_to_global_mapping_matrix)
    
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
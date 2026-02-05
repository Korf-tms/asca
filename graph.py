from scipy.sparse.csgraph import laplacian
from collections import deque
from scipy.sparse import coo_matrix

import pathlib as pl
import pandas as pd
import numpy as np
from collections import Counter

import time

class Vertex:
    """
    Vertex class
    id : int - unique identifier of the vertex
    adj : list - list of adjacent vertices in format (vertex, weight)
    coarse : bool - if tis vertex is coarse
    name : str - name of the vertex for visualization purposes
    graph : Subgraph - subgraph that belongs to the vertex. All the subgraphs are tied to a vertex.
    """
    def __init__(self, id):
        self.id = id
        self.adj = []
        self.coarse = False
        self.name = ""
        self.graph = None
    
    def get_adj(self):
        return [neighbor for neighbor, _ in self.adj]

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
        self.name = "Graph"
    

    """
    Creates vertex list from rows, cols, values (coo_matrix format).
    """
    @staticmethod
    def vertex_list_from_coo(rows, cols, values):

        if len(rows) != len(cols) != len(values):
            raise ValueError("Invalid COO representation")

        n = int(max(rows.max(), cols.max()) + 1)#get highest vertex index
        vertex_dictionary = {i: Vertex(i) for i in range(n)}
        for row, col, val in zip(rows, cols, values):
            vertex_row = vertex_dictionary[int(row)]
            vertex_col = vertex_dictionary[int(col)]
            vertex_row.adj.append((vertex_col, val))

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
        return cls(cls.vertex_list_from_coo(rows, cols, values))

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
                values = dataframe['val'].to_numpy(dtype=np.float64)
                return cls(cls.vertex_list_from_coo(rows, cols, values))
            elif "/adj_matrix" in keys:
                dataframe = store.get("adj_matrix")
                adj_matrix = coo_matrix(dataframe.to_numpy())
                return cls(cls.vertex_list_from_coo(adj_matrix.row, adj_matrix.col, adj_matrix.data))
            else:
                raise ValueError(f"HDF5 file {path} does not contain 'coo_matrix' or 'adj_matrix' key.")              
                 
    @classmethod
    def from_coo(cls, adj_matrix : coo_matrix):
        return cls(cls.vertex_list_from_coo(adj_matrix.row, adj_matrix.col, adj_matrix.data))

    """
    Creates adjacency matrix from given vertex list, the order of vertex list matters.
    """
    def vertex_list_to_adj_matrix(self, vertex_list):
        if not vertex_list:
            return 0

        mapping = {v: i for i, v in enumerate(vertex_list)}

        # Collect all edges using list comprehensions
        row = []
        col = []
        val = []
        for v in vertex_list:
            i = mapping[v]
            row.extend([i] * len(v.adj))
            col.extend(mapping[n] for n, _ in v.adj)
            val.extend(w for _, w in v.adj)
        # Build matrix
        shape = len(vertex_list)
        mat = np.zeros((shape, shape), dtype=np.float64)
        mat[row, col] = val
        return mat


    """
    computes the local schur complement for the graph
    """
    def schur_complement(self, num_coarse, adjacency_matrix):
        if num_coarse == 0:
            print("Warning: no coarse vertices in graph.")
            return 0
        
        adjacency_matrix_laplacian = laplacian(adjacency_matrix, dtype=np.float64)
        a11 = adjacency_matrix_laplacian[0:num_coarse, 0:num_coarse]
        a22 = adjacency_matrix_laplacian[num_coarse:, num_coarse:]
        a21 = adjacency_matrix_laplacian[num_coarse:, 0:num_coarse]
        a12 = adjacency_matrix_laplacian[0:num_coarse, num_coarse:]

        return a11 - (a12 @ np.linalg.inv(a22) @ a21)
    
    def local_schur_complement(self):
        sorted_vertices = sorted(self.vertex_list, key=lambda x: (not x.coarse, x.id), reverse=False)
        adjacency_matrix = self.vertex_list_to_adj_matrix(sorted_vertices)
        return self.schur_complement(len(self.coarse_vertices), adjacency_matrix)

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
            remaining_vertices.difference_update(current.get_adj())
        self.set_coarse(coarse_vertices)
        return coarse_vertices

    """
    Creates subgraphs around each coarse vertex with given depth.
    """
    def create_subgraphs_depth(self, max_depth = 2):
        if max_depth < 1:
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

                for neighbor in current.get_adj():
                    if neighbor in visited:
                        continue
                    if depth[current] + 1 > max_depth + 1:
                        continue
                    visited.add(neighbor)
                    depth[neighbor] = depth[current] + 1
                    queue.append(neighbor)

            vertex.graph = SubGraph(vertex_list=subgraph_vertex_list, graph=self, name=f"SubGraph{iterator}")

class UniversalGraph(Graph):
    """
    Generic graph
    """
    def __init__(self, vertex_list):
        super().__init__(vertex_list)
        self.edge_count = dict()
        self.coarse_vertices = list()

class GridGraph(Graph):
    """
    Grid graph
    """
    def __init__(self, vertex_list):
        super().__init__(vertex_list)
        self.edge_count = dict()
        self.coarse_vertices = list()

    @classmethod
    def from_csv(cls, path):
        base = Graph.from_csv(path)
        return cls(base.vertex_list)

    @classmethod
    def from_hdf5(cls, path):
        base = Graph.from_hdf5(path)
        return cls(base.vertex_list)
    
    @classmethod
    def from_coo(cls, adj_matrix : coo_matrix):
        base = Graph.from_coo(adj_matrix)
        return cls(base.vertex_list)

    """
    
    """
    def select_coarse_moore_neighborhood(self, spacing = 1):
        coarse_vertices = set()
        visited = set()
        for vertex in self.vertex_list:
            if vertex in visited:
                continue
            if vertex not in coarse_vertices:
                coarse_vertices.add(vertex)
                visited.update(self.get_moore_neighborhood(vertex, spacing))

        self.set_coarse(coarse_vertices)
        return coarse_vertices
    
    """
    """
    def select_coarse_neighborhood(self, size = 1):
        coarse_vertices = set()
        visited = set()
        for vertex in self.vertex_list:
            if vertex in visited:
                continue
            if vertex not in coarse_vertices:
                coarse_vertices.add(vertex)
                adj = set([vertex])
                for _ in range(size):
                    adj.update(*(v.get_adj() for v in adj))
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
    Creates the maximum possible number of subgraphs,
    size = 1 means one vertice in each direction.
    """
    def create_subgraphs_all(self, size = 1):

        for iterator, vertex in enumerate(self.vertex_list):

            subgraph_vertex_list = self.get_moore_neighborhood(vertex, size)

            #graphs with less that 3 coarse vertices are not useful
            if len([vertex for vertex in subgraph_vertex_list if vertex.coarse]) < 3:
                continue

            vertex.graph = SubGraph(vertex_list=subgraph_vertex_list, graph=self, name=f"SubGraph{iterator}")

    """
    Helper method that returns vertices around given vertex in square of given size.
    """
    def get_moore_neighborhood(self, vertex, size = 1):
        selected_vertices = set()
        selected_vertices.add(vertex)
        current_layer = set(vertex.get_adj())
        for i in range(size):

            #find all neighbours of loop for finding the corners
            neighbours = []
            for neighbour in current_layer:
                neighbours.extend(neighbour.get_adj())

            #find all vertices that are adjacent to at least one neighbour meaning the corners
            counts = {}
            for item in neighbours:
                if item in selected_vertices:
                    continue
                counts[item] = counts.get(item, 0) + 1
            corners = {key for key in counts.keys() if counts[key] > 1 and key != vertex}

            current_layer.update(corners)

            selected_vertices.update(current_layer)

            if i == size - 1:
                break

            new_layer = set()
            for neighbour in current_layer:
                new_layer.update(neighbour.get_adj())

            current_layer = new_layer
            
        return selected_vertices

class SubGraph(Graph):
    """
    Subgraph class
    Every subgraph is tied to a vertex in the main graph, that vertex acts as a origin of the subgraph.
    """
    def __init__(self, vertex_list, graph, name):
        self.vertex_list = list()
        self.name = name
        self.parent = graph

        original_vertex_to_subgraph_vertex = dict()
        for i, original_vertex in enumerate(vertex_list):#create subgraph vertices
            subgraph_vertex = SubgraphVertex(id= i, vertex=original_vertex) 
            original_vertex_to_subgraph_vertex[original_vertex] = subgraph_vertex
            self.vertex_list.append(subgraph_vertex)

        vertex_list_set = set(vertex_list)

        for vertex in self.vertex_list:#populate adjacency lists for vertices
            vertex.adj.extend([(original_vertex_to_subgraph_vertex[original_vertex], weight) for original_vertex, weight in vertex.original_vertex.adj if original_vertex in vertex_list_set])

        for vertex in self.vertex_list:#populate the edge count in the parent graph, this is needed for overlapping subgraphs
            for neighbour in vertex.get_adj():
                key = (vertex.original_vertex.id, neighbour.original_vertex.id)
                graph.edge_count[key] = graph.edge_count.get(key, 0) + 1

        self.num_coarse = sum(1 for vertex in self.vertex_list if vertex.original_vertex.coarse)
        self.sorted_vertex_list = sorted(self.vertex_list, key=lambda vertex: (not vertex.original_vertex.coarse, vertex.id), reverse=False)
    
    def local_schur_complement(self):
        adjacency_matrix = self.vertex_list_to_adj_matrix(self.sorted_vertex_list)

        vertex_ajd_matrix_mapping = {vertex: iterator for iterator, vertex in enumerate(self.sorted_vertex_list)}
        for vertex in self.vertex_list:
            for neighbour in vertex.get_adj():
                adjacency_matrix[vertex_ajd_matrix_mapping[vertex], 
                                 vertex_ajd_matrix_mapping[neighbour]] /= self.parent.edge_count[(vertex.original_vertex.id, 
                                                                                                    neighbour.original_vertex.id)]
        return self.schur_complement(self.num_coarse, adjacency_matrix)
    
    def local_to_global_mapping(self):
        local_to_global_mapping_matrix = np.zeros((len(self.parent.coarse_vertices), self.num_coarse))#mapping matrix
        coarse = [x for x in self.vertex_list if x.original_vertex.coarse]
        
        for iterator, vertex in enumerate(coarse):
            local_to_global_mapping_matrix[self.parent.sorted_vertex_adj_matrix_mapping[vertex.original_vertex]][iterator] = 1

        return local_to_global_mapping_matrix
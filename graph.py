from collections import deque
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, diags, issparse
from scipy.sparse.linalg import spsolve
from collections import Counter, defaultdict


import pathlib as pl
import pandas as pd
import h5py
import numpy as np
import scipy.io as spio

class Vertex:
    pass
class SubGraph:
    pass

class Edge:

    def __init__(self, start : Vertex, end : Vertex, weight : int):
        self.start : Vertex = start
        self.end : Vertex = end
        self.weight : int = weight
        self.multiplicity : int = 0

    def __hash__(self):
        return hash((self.start, self.end))

    def __eq__(self, other):
        return ( isinstance(other, Edge) and self.start == other.start and self.end == other.end)

class Vertex:
    """
    Vertex class
    id : int - unique identifier of the vertex
    adj : list - list of adjacent vertices in format (vertex, weight)
    coarse : bool - if tis vertex is coarse
    name : str - name of the vertex for visualization purposes
    graph : Subgraph - subgraph that belongs to the vertex. All the subgraphs are tied to a vertex.
    """
    def __init__(self, id : int):
        self.id : int = id
        self.adj : set[Edge] = set()
    
    def get_adj(self) -> set[Vertex]: 
        return {edge.end for edge in self.adj}
    
    def get_in_subgrah(self, subgraph : set[Vertex]) -> set[Vertex]:
        return {edge for edge in self.adj if edge.end in subgraph}

    def __str__(self):
        return f"Vertex: {self.id}"
    
    def __repr__(self):
        return f"Vertex: {self.id}"

    def __hash__(self):
        return self.id
    
    def __eq__(self, other):
        return isinstance(other, Vertex) and self.id == other.id

class Graph:
    """
    Base graph class
    vertex_list : list of Vertex obejects - main representation of the graph
    edge_count : dict - counts how many subgraphs each edge is part of
    coarse_vertices : list - list of coarse vertices in the graph, is populated after one of the select_coarse methods is called
    name : str - name of the graph for visualization purposes
    """
    def __init__(self, 
                 vertex_list : list[Vertex] = None,
                 csr_matrix : csr_matrix = None,
                 path : str = None
                 ):
        self.vertex_list = list()
        self.subgraph_list = list()
        self.edge_count = Counter()
        self.name = "Graph"
        if vertex_list:
            self.vertex_list = vertex_list
        elif csr_matrix != None:
            coo_matrix_obj = csr_matrix.tocoo()
            self.vertex_list = self._vertex_list_from_coo(
                coo_matrix_obj.row,
                coo_matrix_obj.col,
                coo_matrix_obj.data,
            )
        elif path != None:
            path_obj = pl.Path(path)
            if not path_obj.exists():
                raise FileNotFoundError(f"File {path} does not exist.")
            if path_obj.suffix == ".csv":
                self.vertex_list = self._vertex_list_from_csv(path=path)
            elif path_obj.suffix == ".hdf5":
                self.vertex_list = self._vertex_list_from_hdf5(path=path)
            elif path_obj.suffix == ".mat":
                self.vertex_list = self._vertex_list_from_mat(path=path)

    @staticmethod
    def _vertex_list_from_coo(rows, cols, values):
        if not (len(rows) == len(cols) == len(values)):
            raise ValueError("Invalid COO representation")

        n = int(max(max(rows), max(cols)) + 1)#get highest vertex index
        vertex_dictionary = {i: Vertex(i) for i in range(n)}
        for row, col, val in zip(rows, cols, values):
            vertex_row = vertex_dictionary[int(row)]
            vertex_col = vertex_dictionary[int(col)]
            vertex_row.adj.add(Edge(vertex_row, vertex_col, val))

        return list(vertex_dictionary.values())
    
    @staticmethod
    def _vertex_list_from_csv(path):
        dataframe = pd.read_csv(path)
        return Graph._vertex_list_from_coo(dataframe["row"], dataframe["col"], dataframe["val"])

    @staticmethod
    def _vertex_list_from_hdf5(path):
        with h5py.File(path, "r") as file:
            if "coo_matrix" in file:
                group = file["coo_matrix"]
                return Graph._vertex_list_from_coo(group["row"][:], group["col"][:], group["val"][:])
            elif "adj_matrix" in file:
                adj = coo_matrix(file["adj_matrix"])
                return Graph._vertex_list_from_coo(adj.row, adj.col, adj.data)
            else:
                raise ValueError(
                    f"HDF5 file {path} does not contain 'coo_matrix' or 'adj_matrix'."
                )

    @staticmethod
    def _vertex_list_from_mat(path):
        mat = spio.loadmat(path)
        if "Problem" not in mat:
            raise ValueError(f"MAT file {path} does not contain 'Problem' key.")

        adj = mat["Problem"][0][0][1]
        if not hasattr(adj, "indptr"):
            adj = mat["Problem"][0][0][2]

        coo_adj = adj.tocoo()
        return Graph._vertex_list_from_coo(coo_adj.row, coo_adj.col, coo_adj.data)
    
    """
    Loads adjacency matrix from .mtx file, handles both sparse and dense case,
    although very crudely.
    """
    @staticmethod  
    def _vertex_list_from_mtx(path):
        adj_mat = spio.mmread(path)
       
        # Handle both sparse and dense matrices
        if issparse(adj_mat):
            # Convert to COO format if not already
            coo_adj_mat = adj_mat.tocoo() if not isinstance(adj_mat, coo_matrix) else adj_mat
        else:
            # Dense matrix -> convert to sparse COO format
            coo_adj_mat = coo_matrix(adj_mat)
            
        return Graph._vertex_list_from_coo(coo_adj_mat.row, coo_adj_mat.col,coo_adj_mat.data)


    """
    Sets coarse vertices and creates mapping, that is used in schur complement by the subgraphs
    """
    def set_coarse(self, coarse_vertices):
        self.coarse_vertices = coarse_vertices
        self.coarse_vertices_count = len(coarse_vertices)
        self.sorted_vertex_adj_matrix_mapping = {vertex: i for i, vertex in enumerate(sorted(self.vertex_list, key=self.vertice_sort, reverse=False))}
    
    """
    returns list of subgraphs in the graph
    """

    def vertice_sort(self, vertex):
        return (not vertex in self.coarse_vertices, vertex.id)
    
    """
    Selects coarse vertices that are part of maximal independent set.
    Maximal independent set is a set of vertices such that no two vertices are adjacent.
    And no additional vertices can be added to the set without violating this property.
    @return coarse_vertices - set of coarse vertices
    """
    def select_coarse_mis(self, size = 1):
        self.set_coarse(self.get_mis_set(self.vertex_list, size))
        return self.coarse_vertices

    def vertex_list_to_adj_matrix(self):
        vertex_list = sorted(self.vertex_list, key=self.vertice_sort)
        neighbor_set = set(vertex_list)
        mapping = {v: i for i, v in enumerate(vertex_list)}

        row = []
        col = []
        val = []
        for vertex in vertex_list:
            row_idx = mapping[vertex]
            for edge in vertex.adj:
                if edge.end not in neighbor_set:
                    continue
                row.append(row_idx)
                col.append(mapping[edge.end])
                val.append(edge.weight)

        shape = len(vertex_list)
        mat = csr_matrix((val, (row, col)), shape=(shape, shape), dtype=np.float64)
        return mat

    def get_mis_set(self, vertex_list, size):
        mis_set = set()
        remaining_vertices = set(vertex_list)

        while remaining_vertices:
            current = remaining_vertices.pop()
            mis_set.add(current)
            remaining_vertices.difference_update(self.get_neighbourhood(current, size=size))
        return mis_set

    def select_coarse_moore_neighborhood(self, size = 1):
        coarse_vertices = set()
        visited = set()
        
        for vertex in self.vertex_list:
            if vertex in visited:
                continue
            
            coarse_vertices.add(vertex)
            visited.update(self.get_neighborhood_by_connectivity(vertex, size))
                
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
    Creates subgraphs around each coarse vertex with given depth.
    """
    def get_moore_subgraph(self, vertices, size):
        for vertex in vertices:
            degree = len(vertex.get_adj())
            if degree <= 4:
                yield vertex, self.get_neighborhood_by_connectivity(vertex, size)
            else:
                yield vertex, self.get_neighbourhood(vertex, size=size)
                
    def create_subgraphs_moore_neighborhood_around_coarse(self, size=1):
        for iterator, (vertex, subgraph) in enumerate(self.get_moore_subgraph(self.coarse_vertices, size)):
            if len([vertex for vertex in subgraph if vertex in self.coarse_vertices]) < 3:
                continue
            self.subgraph_list.append(SubGraph(vertex_list=subgraph, graph=self, name=f"SubGraph{iterator}"))

            edges = set()
            for v in subgraph:
                edges.update(v.get_in_subgrah(subgraph))
            for edge in edges:
                edge.multiplicity += 1
    """
    Creates the maximum possible number of subgraphs,
    size = 1 means one vertice in each direction.
    """
    def create_subgraphs_moore_neighborhood_all(self, size=1):
        for iterator, (vertex, subgraph) in enumerate(self.get_moore_subgraph(self.vertex_list, size)):
            if len([vertex for vertex in subgraph if vertex in self.coarse_vertices]) < 3:
                continue
            self.subgraph_list.append(SubGraph(vertex_list=subgraph, graph=self, name=f"SubGraph{iterator}"))

            edges = set()
            for v in subgraph:
                edges.update(v.get_in_subgrah(subgraph))
            for edge in edges:
                edge.multiplicity += 1
    def get_depth_subgraph(self, vertices, max_depth):
        for vertex in vertices:
            yield vertex, self.get_neighbourhood(vertex, size=max_depth)
    """
    Creates subgraphs around each coarse vertex with given depth.
    """
    def create_subgraphs_depth(self, max_depth = 2):
        if max_depth < 1:
            raise ValueError("Max depth must be at least 1.")

        for iterator, (vertex, subgraph) in enumerate(self.get_depth_subgraph(self.coarse_vertices, max_depth)):
            self.subgraph_list.append(SubGraph(vertex_list=subgraph, graph=self, name=f"SubGraph{iterator}"))

            edges = set()
            for v in subgraph:
                edges.update(v.get_in_subgrah(subgraph))
            for edge in edges:
                edge.multiplicity += 1

    
    def create_subgraphs_macrostructures(self, microstructure_size = 2, subgraph_structure_connectivity = 2, macrostructure_microstructure_inclusion_distance = 1):
        subgraphs = dict()
        subgraph_structure_mapping = {vertex : Vertex(vertex.id) for vertex in self.coarse_vertices}

        for vertex in self.coarse_vertices:
            subgraphs[subgraph_structure_mapping[vertex]] = set(self.get_neighbourhood(vertex, microstructure_size))
            visited = set({vertex})
            depth = defaultdict(lambda: 1000)
            depth[vertex] = 0
            queue = deque([vertex])
            
            while queue:
                current = queue.popleft()
                if current in self.coarse_vertices:
                    subgraph_structure_mapping[vertex].adj.add(Edge(subgraph_structure_mapping[vertex], subgraph_structure_mapping[current], 1))
                for neighbor in current.get_adj():
                    if depth[current] + 1 < depth[neighbor]:
                        depth[neighbor] = depth[current] + 1
                    
                    if depth[current] >= subgraph_structure_connectivity or neighbor in visited:
                        continue

                    visited.add(neighbor)
                    queue.append(neighbor)
        macrostructure_centers = self.get_mis_set(subgraph_structure_mapping.values(), 1)

        for i, vertex in enumerate(macrostructure_centers):
            macrostructure = set()
            macrostructure.update(subgraphs[vertex])
            for neighbour in self.get_neighbourhood(vertex, size=macrostructure_microstructure_inclusion_distance):
                macrostructure.update(subgraphs[neighbour])
            self.subgraph_list.append(SubGraph(vertex_list=macrostructure, graph=self, name=f"SubGraph{i}"))
    """
    Returns adjacents vertices of the root vertex to depth of size
    """
    def get_neighbourhood(self, vertex, size = 1):
        selected_vertices = set({vertex})
        for _ in range(size):
            selected_vertices.update(*(v.get_adj() for v in selected_vertices))
        return selected_vertices
    """
    Returns adjacents vertices of the root vertex to depth of size + vertices that have at least size adjacents to the selected vertices
    """
    def get_neighborhood_by_connectivity(self, vertex, size = 1):
        visited = set({vertex})
        depth = dict()
        depth[vertex] = 0
        queue = deque([vertex])
        size += 1

        while queue:
            current = queue.popleft()
            for neighbor in current.get_adj():
                neighbor_depth = depth.get(neighbor, size + 10)
                current_depth = depth.get(current, size + 10)
                
                if current_depth + 1 < neighbor_depth:
                    depth[neighbor] = current_depth + 1
                    neighbor_depth = current_depth + 1

                if neighbor_depth == size and len(set(neighbor.get_adj()).intersection(visited)) <= 1:
                    continue
                
                if current_depth >= size or neighbor in visited:
                    continue
                
                visited.add(neighbor)
                queue.append(neighbor)

        return visited
    
class SubGraph():
    """
    Subgraph class
    Every subgraph is tied to a vertex in the main graph, that vertex acts as a origin of the subgraph.
    """
    def __init__(self, vertex_list, graph, name):
        self.vertex_list : list[Vertex] = vertex_list
        self.name : str = name
        self.parent : Graph = graph

        self.coarse_vertices_count = len([vertex for vertex in vertex_list if vertex in self.parent.coarse_vertices])
        self.sorted_vertex_list = sorted(self.vertex_list, key=self.parent.vertice_sort, reverse=False)
    
    """
    computes the local schur complement for the graph
    """
    def schur_complement(self, num_coarse, adjacency_matrix):
        if num_coarse <= 0:
            raise ValueError("Number of coarse vertices must be greater than 0.")
        
        degrees = np.asarray(adjacency_matrix.sum(axis=1)).ravel()
        graph_laplacian = diags(degrees, format="csr") - adjacency_matrix
        l_11 = graph_laplacian[:num_coarse, :num_coarse]
        l_22 = graph_laplacian[num_coarse:, num_coarse:].tocsc()
        l_21 = graph_laplacian[num_coarse:, :num_coarse].tocsc()
        l_12 = graph_laplacian[:num_coarse, num_coarse:]

        return l_11 - l_12 @ spsolve(l_22, l_21)

    def local_schur_complement(self):
        adjacency_matrix = self.vertex_list_to_adj_matrix(self.sorted_vertex_list)

        schur_complement = self.schur_complement(self.coarse_vertices_count, adjacency_matrix)
        return csr_matrix(schur_complement, dtype=np.float64)
    
    def get_contribution(self):
        mapping = self.local_to_global_mapping()
        schur_complement = self.local_schur_complement()
        temp = mapping @ schur_complement @ mapping.T
        return temp

    def local_to_global_mapping(self):
        coarse = self.sorted_vertex_list[:self.coarse_vertices_count]

        row_ind = list()
        col_ind = list()
    
        mapping = self.parent.sorted_vertex_adj_matrix_mapping
        for iterator, vertex in enumerate(coarse):
            row_ind.append(mapping[vertex])
            col_ind.append(iterator)

        return csr_matrix((np.ones(len(row_ind)), (row_ind, col_ind)), shape=(self.parent.coarse_vertices_count, self.coarse_vertices_count), dtype=np.float64)
    
    """
    Creates adjacency matrix from given vertex list, the order of vertex list matters.
    """
    def vertex_list_to_adj_matrix(self, vertex_list : list[Vertex]):
        if not vertex_list:
            return 0

        neighbor_set = set(vertex_list)
        mapping = {v: i for i, v in enumerate(vertex_list)}

        row = []
        col = []
        val = []
        for vertex in vertex_list:
            row_idx = mapping[vertex]
            for edge in vertex.adj:
                if edge.end not in neighbor_set:
                    continue
                row.append(row_idx)
                col.append(mapping[edge.end])
                val.append(edge.weight / edge.multiplicity)

        shape = len(vertex_list)
        mat = csr_matrix((val, (row, col)), shape=(shape, shape), dtype=np.float64)
        return mat
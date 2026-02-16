from scipy.sparse.csgraph import laplacian
from collections import deque
from scipy.sparse import coo_matrix, csr_matrix
from collections import Counter, defaultdict
from joblib import Parallel, delayed


import pathlib as pl
import pandas as pd
import numpy as np
import scipy.io as spio

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
    def __init__(self, vertex_list : list[Vertex]):
        self.vertex_list = vertex_list
        self.edge_count = Counter()
        self.name = "Graph"
    

    """
    Creates vertex list from rows, cols, values (coo_matrix format).
    """
    @staticmethod
    def vertex_list_from_coo(rows, cols, values):

        if len(rows) != len(cols) != len(values):
            raise ValueError("Invalid COO representation")

        n = int(max(max(rows), max(cols)) + 1)#get highest vertex index
        vertex_dictionary = {i: Vertex(i) for i in range(n)}
        for row, col, val in zip(rows, cols, values):
            vertex_row = vertex_dictionary[int(row)]
            vertex_col = vertex_dictionary[int(col)]
            vertex_row.adj.append((vertex_col, val))

        return list(vertex_dictionary.values())

    @staticmethod
    def vertex_list_from_csr(data, indices, indptr):
        rows = []
        cols = []
        values = []
        for i in range(len(indptr) - 1):
            for j in range(indptr[i], indptr[i + 1]):
                rows.append(i)
                cols.append(indices[j])
                values.append(data[j])
        return Graph.vertex_list_from_coo(rows, cols, values)

    @classmethod
    def from_file(cls, path : str):
        path_obj = pl.Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"File {path} does not exist.")
        if path_obj.suffix == ".csv":
            return cls.from_csv(path_obj)
        elif path_obj.suffix == ".hdf5":
            return cls.from_hdf5(path_obj)
        elif path_obj.suffix == ".mat":
            return cls.from_mat(path_obj)

    """
    Gets coo format from csv file in format row,col,val and creates graph from it.
    """
    @classmethod
    def from_csv(cls, path : pl.Path):
        if path.suffix != ".csv":
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
    def from_hdf5(cls, path : pl.Path):
        if path.suffix != ".hdf5":
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
    def from_mat(cls, path : pl.Path):
        if path.suffix != ".mat":
            raise ValueError(f"File {path} is not a MAT file.")
        
        mat = spio.loadmat(path)
        if 'Problem' not in mat:
            raise ValueError(f"MAT file {path} does not contain 'Problem' key.")
              
        adj_matrix = mat['Problem'][0][0][1]
        try:
            adj_matrix.indptr
        except:
            adj_matrix = mat['Problem'][0][0][2] 
    
        rows = list()
        cols = list()
        values = list()
        for i in range(len(adj_matrix.indptr) - 1):
            for j in range(adj_matrix.indptr[i], adj_matrix.indptr[i + 1]):
                cols.append(i)
                rows.append(adj_matrix.indices[j])
                values.append(adj_matrix.data[j])
        return cls(cls.vertex_list_from_coo(rows, cols, values))

    @classmethod
    def from_coo(cls, adj_matrix : coo_matrix):
        return cls(cls.vertex_list_from_coo(adj_matrix.row, adj_matrix.col, adj_matrix.data))

    @classmethod
    def from_csr(cls, adj_matrix : csr_matrix):
        return cls(cls.vertex_list_from_csr(adj_matrix.data, adj_matrix.indices, adj_matrix.indptr))

    """
    Creates adjacency matrix from given vertex list, the order of vertex list matters.
    """
    def vertex_list_to_adj_matrix(self, vertex_list, divide_edge_weights = False):
        if not vertex_list:
            return 0

        neighbtor_set = set(vertex_list)
        mapping = {v: i for i, v in enumerate(vertex_list)}

        row = []
        col = []
        val = []
        for vertex in vertex_list:
            id = mapping[vertex]
            for neighbor, weight in vertex.adj:
                if neighbor not in neighbtor_set:
                    continue
                row.append(id)
                col.append(mapping[neighbor])
                if divide_edge_weights:
                    val.append(weight / self.edge_count[(vertex.id, neighbor.id)])
                else:
                    val.append(weight)
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
            raise ValueError("Number of coarse vertices must be greater than 0.")
        
        adjacency_matrix_laplacian = laplacian(adjacency_matrix, dtype=np.float64)
        a11 = adjacency_matrix_laplacian[:num_coarse, :num_coarse]
        a22 = adjacency_matrix_laplacian[num_coarse:, num_coarse:]
        a21 = adjacency_matrix_laplacian[num_coarse:, :num_coarse]
        a12 = adjacency_matrix_laplacian[:num_coarse, num_coarse:]

        return a11 - (a12 @ np.linalg.inv(a22) @ a21)
        #return a11 - a12 @ np.linalg.solve(a22, a21)
    
    def local_schur_complement(self):
        sorted_vertices = sorted(self.vertex_list, key=self.vertice_sort, reverse=False)
        adjacency_matrix = self.vertex_list_to_adj_matrix(sorted_vertices)
        return self.schur_complement(len(self.coarse_vertices), adjacency_matrix)

    """
    Sets coarse vertices and creates mapping, that is used in schur complement by the subgraphs
    """
    def set_coarse(self, coarse_vertices):
        for vertex in coarse_vertices:
            vertex.coarse = True
        self.coarse_vertices = coarse_vertices
        self.coarse_vertices_count = len(coarse_vertices)
        self.sorted_vertex_adj_matrix_mapping = {vertex: i for i, vertex in enumerate(sorted(self.vertex_list, key=self.vertice_sort, reverse=False))}
    
    def vertice_sort(self, vertex):
        return (not vertex.coarse, vertex.id)

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
    def select_coarse_mis(self, size = 1):
        coarse_vertices = set()
        remaining_vertices = set(self.vertex_list)

        while remaining_vertices:
            current = remaining_vertices.pop()
            coarse_vertices.add(current)
            remaining_vertices.difference_update(self.get_neighbourhood(current, size=size))
        self.set_coarse(coarse_vertices)
        return coarse_vertices

    """
    Creates subgraphs around each coarse vertex with given depth.
    """
    def create_subgraphs_depth(self, max_depth = 2):
        if max_depth < 1:
            raise ValueError("Max depth must be at least 1.")

        for iterator, vertex in enumerate(self.coarse_vertices):
            vertex_list, edge_list = self.get_neighbourhood_with_edges(vertex, size=max_depth)
            self.edge_count.update(edge_list)
            vertex.graph = SubGraph(
                vertex_list=vertex_list, 
                graph=self, 
                name=f"SubGraph{iterator}"
            )

    def get_neighbourhood_with_edges(self, vertex, size = 1):
        visited = set({vertex})
        keys = set()
        depth = defaultdict(lambda: 1000)
        depth[vertex] = 0
        queue = deque([vertex])
        
        while queue:
            current = queue.popleft()
            for neighbor in current.get_adj():
                if depth[current] + 1 < depth[neighbor]:
                    depth[neighbor] = depth[current] + 1

                if depth[neighbor] <= size:
                    keys.add((current.id, neighbor.id))
                
                if depth[current] >= size or neighbor in visited:
                    continue

                visited.add(neighbor)
                queue.append(neighbor)

        return (list(visited), keys)

    def get_neighbourhood(self, vertex, size = 1):
        selected_vertices = set({vertex})
        for _ in range(size):
            selected_vertices.update(*(v.get_adj() for v in selected_vertices))
        return list(selected_vertices)
    
class UniversalGraph(Graph):
    """
    Generic graph
    """
    def __init__(self, vertex_list):
        super().__init__(vertex_list)
        self.coarse_vertices = list()

class GridGraph(Graph):
    """
    Grid graph
    """
    def __init__(self, vertex_list):
        super().__init__(vertex_list)
        self.coarse_vertices = list()
    """
    """
    def select_coarse_moore_neighborhood(self, spacing = 1):
        coarse_vertices = set()
        visited = set()
        
        for vertex in self.vertex_list:
            if vertex in visited:
                continue
            
            coarse_vertices.add(vertex)
            visited.update(self.get_neighborhood_by_connectivity(vertex, spacing)[0])
                
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
    def create_subgraphs_moore_neighborhood_around_coarse(self, size=1):
        for iterator, vertex in enumerate(self.coarse_vertices):
            degree = len(vertex.get_adj())
            keys = set()
            if degree <= 4:
                subgraph_vertex_list, keys = self.get_neighborhood_by_connectivity(vertex, size)
            
            else:
                subgraph_vertex_list, keys = self.get_neighbourhood_with_edges(vertex, size=size)
            self.edge_count.update(keys)
            #graphs with less that 3 coarse vertices are not useful
            if len([vertex for vertex in subgraph_vertex_list if vertex.coarse]) < 3:
                continue

            vertex.graph = SubGraph(vertex_list=subgraph_vertex_list, graph=self, name=f"SubGraph{iterator}")
    """
    Creates the maximum possible number of subgraphs,
    size = 1 means one vertice in each direction.
    """
    def create_subgraphs_moore_neighborhood_all(self, size = 1):
        for iterator, vertex in enumerate(self.vertex_list):
            degree = len(vertex.get_adj())
            keys = set()
            if degree <= 4:
                subgraph_vertex_list, keys = self.get_neighborhood_by_connectivity(vertex, size)
            
            else:
                subgraph_vertex_list, keys = self.get_neighbourhood_with_edges(vertex, size=size)
            self.edge_count.update(keys)
            #graphs with less that 3 coarse vertices are not useful
            if len([vertex for vertex in subgraph_vertex_list if vertex.coarse]) < 3:
                continue

            vertex.graph = SubGraph(vertex_list=subgraph_vertex_list, graph=self, name=f"SubGraph{iterator}")

    """
    Selects adjacent vertices to root vertex + selects vertices that have 2 or more adjacents of the last layer of selected vertices, in the selectet set
    """
    def get_neighborhood_by_connectivity(self, vertex, size = 1):
        visited = set({vertex})
        keys = set()
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

                if neighbor_depth <= size:
                    keys.add((current.id, neighbor.id))
                
                if current_depth >= size or neighbor in visited:
                    continue
                
                visited.add(neighbor)
                queue.append(neighbor)

        return (list(visited), keys)

class SubGraph(Graph):
    """
    Subgraph class
    Every subgraph is tied to a vertex in the main graph, that vertex acts as a origin of the subgraph.
    """
    def __init__(self, vertex_list, graph, name):
        self.vertex_list = vertex_list
        self.name = name
        self.parent = graph

        #each subgraph has different coarse vertice count, but we already iterate through the vertex list before creation so we could count the coarse vertices there
        self.coarse_vertices_count = len([vertex for vertex in vertex_list if vertex.coarse])
        self.sorted_vertex_list = sorted(self.vertex_list, key=self.parent.vertice_sort, reverse=False)
    
    def local_schur_complement(self):
        #needed
        adjacency_matrix = self.parent.vertex_list_to_adj_matrix(self.sorted_vertex_list, divide_edge_weights=True)

        schur_complement = self.schur_complement(self.coarse_vertices_count, adjacency_matrix)
        return csr_matrix(schur_complement, dtype=np.float64)
    
    def local_to_global_mapping(self):
        coarse = self.sorted_vertex_list[:self.coarse_vertices_count]

        row_ind = list()
        col_ind = list()
    
        mapping = self.parent.sorted_vertex_adj_matrix_mapping
        for iterator, vertex in enumerate(coarse):
            row_ind.append(mapping[vertex])
            col_ind.append(iterator)

        return csr_matrix((np.ones(len(row_ind)), (row_ind, col_ind)), shape=(self.parent.coarse_vertices_count, self.coarse_vertices_count), dtype=np.float64)
from __future__ import annotations

from scipy.sparse import csr_matrix
from numpy import float64


class Edge:
    """
    Class representing an unordered edge connecting tow vertices.

    Attributes
    ----------
    vertices : set[Vertex]

    """

    def __init__(self, first: Vertex, second: Vertex, weight: int):
        if first == second:
            raise ValueError("Self loops are not supported.")

        self.first = first
        self.second = second
        self.weight = weight
        self.multiplicity = 0
        self._unique = frozenset((self.first, self.second))

    def __hash__(self):
        return hash(self._unique)

    def __eq__(self, other):
        return isinstance(other, Edge) and self._unique == other._unique

    def get_other(self, vertex: Vertex) -> Vertex:
        if vertex == self.first:
            return self.second
        if vertex == self.second:
            return self.first
        raise ValueError("Vertex not in edge.")


class Vertex:
    def __init__(self, id: int):
        self.id: int = id
        self.adj: set[Edge] = set()

    def get_adj(self) -> set[Vertex]:
        return {edge.get_other(self) for edge in self.adj}

    def get_in_subgraph(self, subgraph: set[Vertex]) -> set[Edge]:
        return {edge for edge in self.adj if edge.get_other(self) in subgraph}

    def __str__(self):
        return f"Vertex: {self.id}"

    def __repr__(self):
        return f"Vertex: {self.id}"

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return isinstance(other, Vertex) and self.id == other.id


class Graph:
    def __init__(self, vertex_list: set[Vertex] | list[Vertex], name: str = "Graph"):
        self.vertex_list: set[Vertex] = set(vertex_list)
        self.name = name

    def __repr__(self):
        return self.name

    def to_adj_matrix(self, divide_edges=False, sorting=None) -> csr_matrix:
        vertex_list = []
        if sorting is None:
            vertex_list = self.vertex_list
        else:
            vertex_list = sorted(self.vertex_list, key=sorting)
        neighbor_set = set(vertex_list)
        mapping = {v: i for i, v in enumerate(vertex_list)}

        row = []
        col = []
        val = []
        for vertex in vertex_list:
            row_idx = mapping[vertex]
            for edge in vertex.adj:
                connected = edge.get_other(vertex)
                if connected not in neighbor_set:
                    continue
                row.append(row_idx)
                col.append(mapping[connected])
                val.append(
                    edge.weight if not divide_edges else edge.weight / edge.multiplicity
                )

        shape = len(vertex_list)
        mat = csr_matrix((val, (row, col)), shape=(shape, shape), dtype=float64)
        return mat


class OriginalGraph(Graph):

    def __init__(self, vertex_list: set[Vertex] | list[Vertex], name="OriginalGraph"):
        super().__init__(vertex_list, name)
        self.coarse_vertices: set[Vertex] = []
        self.subgraph_list: set[SubGraph] = []

    def set_coarse(self, coarse_vertices: set[Vertex]):
        self.coarse_vertices = coarse_vertices
        self.coarse_vertices_count = len(coarse_vertices)
        self.sorted_vertex_adj_matrix_mapping = {
            vertex: i
            for i, vertex in enumerate(
                sorted(self.vertex_list, key=self.vertex_sort, reverse=False)
            )
        }

    def add_subgraph(self, subgraph_vertices, name: str = None) -> None:
        subgraph = SubGraph(vertex_list=subgraph_vertices, graph=self, name=name)
        self.subgraph_list.append(subgraph)

        edges = set()
        for v in subgraph_vertices:
            edges.update(v.get_in_subgraph(subgraph_vertices))
        for edge in edges:
            edge.multiplicity += 1

    def vertex_sort(self, vertex):
        return (vertex not in self.coarse_vertices, vertex.id)


class SubGraph(Graph):
    def __init__(
        self,
        vertex_list: list[Vertex] | set[Vertex],
        graph: OriginalGraph,
        name="Subgraph",
    ):
        super().__init__(vertex_list, name)
        self.parent = graph

        self.coarse_vertices_count = len(
            [vertex for vertex in vertex_list if vertex in self.parent.coarse_vertices]
        )
        self.sorted_vertex_list = sorted(
            self.vertex_list, key=self.parent.vertex_sort, reverse=False
        )

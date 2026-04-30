from __future__ import annotations

from scipy.sparse import csr_matrix
from numpy import float64


class Edge:
    """
    Representation of edge between two distinct vertices.

    Parameters
    ----------
    first : Vertex
        Source vertex for this edge.
    second : Vertex
        Target vertex for this edge.
    weight : int
        Weight (or value) associated with the edge.

    Attributes
    ----------
    first : Vertex
        Source vertex.
    second : Vertex
        Target vertex.
    weight : int
        Edge weight.
    multiplicity : int
        Counter used to track how many subgraphs this edge is in.
    """

    def __init__(self, first: Vertex, second: Vertex, weight: int):
        if first == second:
            raise ValueError("Self loops are not supported.")

        self.first = first
        self.second = second
        self.weight = weight
        self.multiplicity = 0
        self._unique = (self.first, self.second)

    def __hash__(self):
        return hash(self._unique)

    def __eq__(self, other):
        return isinstance(other, Edge) and self._unique == other._unique

    def get_other(self) -> Vertex:
        """
        Return the target endpoint of the edge.

        Returns
        -------
        Vertex
            The target endpoint of the edge.
        """
        return self.second


class Vertex:
    """
    Representation of a graph vertex.

    Parameters
    ----------
    id : int
        Unique identifier of the vertex.

    Attributes
    ----------
    id : int
        Vertex identifier.
    adj : set[Edge]
        Set of edges incident to this vertex.
    """

    def __init__(self, id: int):
        self.id: int = id
        self.adj: set[Edge] = set()

    def get_adj(self) -> set[Vertex]:
        """
        Get all adjacent vertices.

        Returns
        -------
        set[Vertex]
            Set of vertices connected to this vertex by an edge.
        """
        return {edge.get_other() for edge in self.adj}

    def get_in_subgraph(self, subgraph: set[Vertex]) -> set[Edge]:
        """
        Get edges of this vertex that are in a subgraph.

        Parameters
        ----------
        subgraph : set[Vertex]
            Set of vertices defining the subgraph.

        Returns
        -------
        set[Edge]
            Set of edges in a subgraph
        """
        if self not in subgraph:
            raise ValueError("Vertex not in subgraph.")
        return {edge for edge in self.adj if edge.get_other() in subgraph}

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
    Base class representing an undirected graph.

    Parameters
    ----------
    vertex_list : set[Vertex] or list[Vertex]
        Collection of vertices in the graph.
    name : str, default="Graph"
        Name of the graph.

    Attributes
    ----------
    vertex_list : set[Vertex]
        Set of vertices in the graph.
    name : str
        Graph name.
    """

    def __init__(self, vertex_list: set[Vertex] | list[Vertex], name: str = "Graph"):
        self.vertex_list: set[Vertex] = set(vertex_list)
        self.name = name

    def __repr__(self):
        return self.name

    def to_adj_matrix(self, divide_edges=False, sorting=None) -> csr_matrix:
        """
        Convert the graph to a CSR adjacency matrix.

        Parameters
        ----------
        divide_edges : bool, default=False
            If True, divide edge weights by their multiplicity.
        sorting : callable, optional
            Function used to sort vertices before matrix construction.

        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse adjacency matrix of shape (n_vertices, n_vertices).
        """
        vertex_list = []
        if sorting is None:
            vertex_list = self.vertex_list
        else:
            vertex_list = list(sorted(self.vertex_list, key=sorting))
        neighbor_set = set(vertex_list)
        mapping = {v: i for i, v in enumerate(vertex_list)}

        row = []
        col = []
        val = []
        for vertex in vertex_list:
            row_idx = mapping[vertex]
            for edge in vertex.adj:
                if edge.second not in neighbor_set:
                    continue
                row.append(row_idx)
                col.append(mapping[edge.second])
                val.append(
                    edge.weight if not divide_edges else edge.weight / edge.multiplicity
                )

        shape = len(vertex_list)
        mat = csr_matrix((val, (row, col)), shape=(shape, shape), dtype=float64)
        return mat


class OriginalGraph(Graph):
    """
    Extension of Graph that supports coarse vertices and subgraphs.

    Parameters
    ----------
    vertex_list : set[Vertex] or list[Vertex]
        Collection of vertices.
    name : str, default="OriginalGraph"
        Name of the graph.

    Attributes
    ----------
    coarse_vertices : set[Vertex]
        Set of coarse vertices.
    subgraph_list : set[SubGraph]
        Set of subgraphs derived from this graph.
    """

    def __init__(self, vertex_list: set[Vertex] | list[Vertex], name="OriginalGraph"):
        super().__init__(vertex_list, name)
        self.coarse_vertices: set[Vertex] = set()
        self.subgraph_list: set[SubGraph] = set()

    def set_coarse(self, coarse_vertices: set[Vertex]):
        """
        Defince coarse vertices and create position of vertex on the final approximation matrix.

        Parameters
        ----------
        coarse_vertices : set[Vertex]
            Vertices to mark as coarse.
        """
        self.coarse_vertices = coarse_vertices
        self.coarse_vertices_count = len(coarse_vertices)
        self.sorted_vertex_adj_matrix_mapping = {
            vertex: i
            for i, vertex in enumerate(
                sorted(self.vertex_list, key=self.vertex_sort, reverse=False)
            )
        }

    def add_subgraph(
        self, subgraph_vertices: list[Vertex] | set[Vertex], name: str = None
    ) -> None:
        """
        Add a subgraph.

        Parameters
        ----------
        subgraph_vertices : list[Vertex] or set[Vertex]
            Vertices forming the subgraph.
        name : str, optional
            Name of the subgraph.
        """
        subgraph = SubGraph(vertex_list=set(subgraph_vertices), graph=self, name=name)
        self.subgraph_list.add(subgraph)

    def update_edge_multiplicities(self) -> None:
        subgraph_membership = {vertex: set() for vertex in self.vertex_list}

        for index, subgraph in enumerate(self.subgraph_list):
            for vertex in subgraph.vertex_list:
                subgraph_membership[vertex].add(index)

        for vertex in self.vertex_list:
            for edge in vertex.adj:
                edge.multiplicity = len(
                    subgraph_membership[edge.first].intersection(
                        subgraph_membership[edge.second]
                    )
                )

    def remove_vertex(self, vertex: Vertex):
        for neighbor in vertex.get_adj():
            ro_remove = set()
            for edge in neighbor.adj:
                if edge.first == vertex or edge.second == vertex:
                    ro_remove.add(edge)
            neighbor.adj.difference_update(ro_remove)
        self.vertex_list.remove(vertex)

    def vertex_sort(self, vertex):
        """
        Sorting key for vertices.

        Coarse vertices are placed first, then sorted by vertex id.

        Parameters
        ----------
        vertex : Vertex

        Returns
        -------
        tuple
            Sorting key.
        """
        return (vertex not in self.coarse_vertices, vertex.id)


class SubGraph(Graph):
    """
    Representation of a subgraph derived from an OriginalGraph.

    Parameters
    ----------
    vertex_list : list[Vertex] or set[Vertex]
        Vertices in the subgraph.
    graph : OriginalGraph
        Parent graph.
    name : str, default="Subgraph"
        Name of the subgraph.

    Attributes
    ----------
    parent : OriginalGraph
        Reference to the original graph.
    coarse_vertices_count : int
        Number of coarse vertices in this subgraph.
    sorted_vertex_list : list[Vertex]
        Vertices sorted according to parent graph sort.
    """

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
        self.sorted_vertex_list = list(
            sorted(self.vertex_list, key=self.parent.vertex_sort, reverse=False)
        )

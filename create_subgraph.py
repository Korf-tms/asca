from collections import defaultdict, deque

from graph import Vertex, Edge, OriginalGraph
from graphutils import get_moore_neighborhood, get_mis_set, get_neighborhood


def _get_moore_subgraph(vertices: set[Vertex] | list[Vertex], size: int):
    """
    Generator that yields moore neighborhoods of size for each vertex in vertices.

    Parameters
    ----------
    vertices : iterable of Vertex
        Vertices to generate subgraphs around.
    size : int
        Neighborhood size.

    Yields
    ------
    set[Vertex]
        A set of vertices for each input vertex.
    """
    for vertex in vertices:
        yield get_moore_neighborhood(vertex, size)


def _get_depth_subgraph(vertices, size):
    """
    Generator that yields neighborhood of size for each vertex in vertices.

    Parameters
    ----------
    vertices : iterable of Vertex
        Vertices to generate subgraphs around.
    size : int
        Maximum neighborhood depth, inclusive.

    Yields
    ------
    set[Vertex]
        A subgraph for each vertex consisting of all vertices within
        max_depth distance.
    """
    for vertex in vertices:
        yield get_neighborhood(vertex, size=size)


def moore_neighborhood_around_coarse(graph: OriginalGraph, size: int = 2):
    """
    Create subgraphs around coarse vertices using Moore neighborhoods.

    Parameters
    ----------
    graph : OriginalGraph
        Graph containing coarse vertices.
    size : int, default=2
        Neighborhood size used for subgraph construction, inclusive.
    """
    for iterator, subgraph in enumerate(
        _get_moore_subgraph(graph.coarse_vertices, size)
    ):
        graph.add_subgraph(subgraph, f"SubGraph{iterator}")


def moore_neighborhood_all(graph: OriginalGraph, size: int = 2):
    """
    Create subgraphs for all vertices using Moore neighborhood.

    Parameters
    ----------
    graph : OriginalGraph
        Graph containing vertices and coarse vertex information.
    size : int, default=2
        Neighborhood size used for subgraph construction, inclusive.
    """
    for iterator, subgraph in enumerate(_get_moore_subgraph(graph.vertex_list, size)):
        graph.add_subgraph(subgraph, f"SubGraph{iterator}")


def create_subgraphs_depth(graph: OriginalGraph, size: int = 2):
    """
    Create subgraphs around coarse vertices, each subgeraph is a neighborhood of size.

    Parameters
    ----------
    graph : OriginalGraph
        Graph containing coarse vertices.
    size : int, default=2
        Neighborhood size used for subgraph construction, inclusive.
    """
    for iterator, subgraph in enumerate(
        _get_depth_subgraph(graph.coarse_vertices, size)
    ):
        graph.add_subgraph(subgraph, f"SubGraph{iterator}")


def create_subgraphs_macrostructures(
    graph: OriginalGraph,
    micro_size=2,
    connection_depth=2,
    merge_distance=1,
):
    """
    Create subgraphs by grouping neighborhoods of coarse vertices.

    For each coarse vertex, a local neighborhood is built (microstructure). The coarse vertices are then
    connected based on distance (subgraph structure), and a maximal independent set of them is selected.
    Around each vertice in this set a new subgraph is created based on distance (macrostructure).
    All microstructures that belong to one macrostructure are merged and make up the final subgraphs.

    Parameters
    ----------
    graph : OriginalGraph
        Input graph with coarse vertices defined.
    micro_size : int, default=2
        Size of neighborhood around each coarse vertex (microstructure).
    connection_depth : int, default=2
        Max distance for connecting coarse vertices (sugbraph structure).
    merge_distance : int, default=1
        Distance for merging neighborhoods into final subgraphs (macrostructure).
    """
    # Map original coarse vertices to structure vertices
    coarse_to_structure = {v: Vertex(v.id) for v in graph.coarse_vertices}

    # Build local neighborhoods (microsubgraphs)
    local_subgraphs = {
        coarse_to_structure[v]: set(get_neighborhood(v, micro_size))
        for v in graph.coarse_vertices
    }

    # Connect coarse vertices into subgraph structrue
    for start_vertex in graph.coarse_vertices:
        structure_vertex = coarse_to_structure[start_vertex]

        visited = {start_vertex}
        depth = defaultdict(lambda: float("inf"))
        depth[start_vertex] = 0
        queue = deque([start_vertex])

        while queue:
            current = queue.popleft()

            if current in graph.coarse_vertices:
                structure_vertex.adj.add(
                    Edge(
                        structure_vertex,
                        coarse_to_structure[current],
                        1,
                    )
                )

            for neighbor in current.get_adj():
                if depth[current] + 1 < depth[neighbor]:
                    depth[neighbor] = depth[current] + 1

                if depth[current] >= connection_depth or neighbor in visited:
                    continue

                visited.add(neighbor)
                queue.append(neighbor)

    # Select coarse vertices of the subgraph structure
    selected_centers = get_mis_set(coarse_to_structure.values(), 1)

    # Build final subgraphs by merging microstructures in each macrostructure
    for i, center in enumerate(selected_centers):
        merged_subgraph = set(local_subgraphs[center])

        for neighbor in get_neighborhood(center, size=merge_distance):
            merged_subgraph.update(local_subgraphs[neighbor])

        graph.add_subgraph(merged_subgraph, f"SubGraph{i}")

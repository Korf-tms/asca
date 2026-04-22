import logging

from .graph import Vertex, Edge, OriginalGraph
from .graphutils import get_moore_neighborhood, get_mis_set, get_neighborhood


logger = logging.getLogger(__name__)


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
        A subgraph for each input vertex consisting of all vertices within
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

    logging.info("Using moore around coarse method")

    if size < 1:
        raise ValueError("size must be at least 1")

    skipped = 0
    not_skipped = 0
    for iterator, subgraph in enumerate(
        _get_moore_subgraph(graph.coarse_vertices, size)
    ):
        if len(subgraph & graph.coarse_vertices) < 3:
            skipped += 1
            continue
        not_skipped += 1
        graph.add_subgraph(subgraph, f"SubGraph{iterator}")

    if skipped != 0:
        logging.warning(
            f"{100 * skipped / (not_skipped + skipped)} subgraphs were skipped."
        )

    if not_skipped == 0:
        raise ValueError(
            "No subgraphs were created because every Moore neighborhood had fewer than 3 coarse vertices"
        )
    logging.info(f"Number of subgraphs created: {not_skipped + skipped}")


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
    logging.info("Using moore neighborhood all method")

    if size < 1:
        raise ValueError("size must be at least 1")

    skipped = 0
    not_skipped = 0
    for iterator, subgraph in enumerate(_get_moore_subgraph(graph.vertex_list, size)):
        if len(subgraph & graph.coarse_vertices) < 3:
            skipped += 1
            continue
        not_skipped += 1
        graph.add_subgraph(subgraph, f"SubGraph{iterator}")

    if skipped != 0:
        logging.warning(
            f"{100 * skipped / (not_skipped + skipped)} subgraphs were skipped."
        )

    if not_skipped == 0:
        raise ValueError(
            "No subgraphs were created because every Moore neighborhood had fewer than 3 coarse vertices"
        )
    logging.info(f"Number of subgraphs created: {not_skipped + skipped}")


def depth(graph: OriginalGraph, size: int = 2):
    """
    Create subgraphs around coarse vertices, each subgeraph is a neighborhood of size.

    Parameters
    ----------
    graph : OriginalGraph
        Graph containing coarse vertices.
    size : int, default=2
        Neighborhood size used for subgraph construction, inclusive.
    """
    logging.info("Using depth subgraphs method")

    if size < 1:
        raise ValueError("size must be at least 1")

    skipped = 0
    not_skipped = 0
    for iterator, subgraph in enumerate(
        _get_depth_subgraph(graph.coarse_vertices, size)
    ):
        if len(subgraph & graph.coarse_vertices) < 3:
            skipped += 1
            continue
        graph.add_subgraph(subgraph, f"SubGraph{iterator}")
        not_skipped += 1

    if skipped != 0:
        logging.warning(
            f"{100 * skipped / (not_skipped + skipped)}% subgraphs were skipped."
        )

    if not_skipped == 0:
        raise ValueError(
            "No subgraphs were created because the graph has no coarse vertices"
        )
    logging.info(f"Number of subgraphs created: {not_skipped + skipped}")


def macrostructures(
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
    logging.info("Using macrostructures method")

    for name, size in [
        ("micro_size", micro_size),
        ("connection_depth", connection_depth),
        ("merge_distance", merge_distance),
    ]:
        if size < 1:
            raise ValueError(f"{name} must be at least 1")

    # We need to create the 'subgraph structure', which is graph made of coarse vertices

    # Creating the 'subgraph strucutre' vertices
    subgraph_structure_vertices = {v: Vertex(v.id) for v in graph.coarse_vertices}

    # The vertices in the 'subgraph structure' are not connected, we need to connect them
    for start_vertex in graph.coarse_vertices:
        structure_vertex = subgraph_structure_vertices[start_vertex]

        possible_neighbors = get_neighborhood(start_vertex, connection_depth)

        for vertex in possible_neighbors:
            if vertex not in graph.coarse_vertices or vertex == start_vertex:
                continue
            structure_vertex_neighbor = subgraph_structure_vertices[vertex]
            structure_vertex.adj.add(
                Edge(structure_vertex, structure_vertex_neighbor, 1)
            )

    # Select coarse vertices of the subgraph structure
    selected_centers = get_mis_set(subgraph_structure_vertices.values(), 1)

    # Build 'microstructures' for each voarse vertex, which serves as its center
    microstructures = {
        subgraph_structure_vertices[v]: set(get_neighborhood(v, micro_size))
        for v in graph.coarse_vertices
    }

    skipped = 0
    not_skipped = 0

    for i, center in enumerate(selected_centers):
        merged_subgraph = set(microstructures[center])

        for neighbor in get_neighborhood(center, size=merge_distance):
            if neighbor in microstructures:
                merged_subgraph.update(microstructures[neighbor])
        if len(merged_subgraph & graph.coarse_vertices) < 3:
            skipped += 1
            continue
        graph.add_subgraph(merged_subgraph, f"SubGraph{i}")
        not_skipped += 1

    if skipped != 0:
        logging.warning(
            f"{100 * skipped / (not_skipped + skipped)}% subgraphs were skipped."
        )

    if not_skipped == 0:
        raise ValueError(
            "No subgraphs were created because no macrostructure centers were selected"
        )
    logging.info(f"Number of subgraphs created: {not_skipped + skipped}")

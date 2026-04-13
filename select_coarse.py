from graph import OriginalGraph, Vertex
from graphutils import get_mis_set, get_moore_neighborhood


def select_coarse_mis(graph: OriginalGraph, size: int = 1) -> set[Vertex]:
    """
    Select coarse vertices using a maximal independent set (MIS).

    Parameters
    ----------
    graph : OriginalGraph
        Input graph.
    size : int, default=1
        Neighborhood size used when computing the MIS, inclusive.

    Returns
    -------
    set[Vertex]
        Selected coarse vertices.
    """
    mis = get_mis_set(graph.vertex_list, size)
    graph.set_coarse(mis)
    return mis


def select_coarse_moore(graph: OriginalGraph, size: int = 1) -> set[Vertex]:
    """
    Select coarse vertices using independent set with moore neighborhoods.

    Parameters
    ----------
    graph : OriginalGraph
        Input graph.
    size : int, default=1
        Size of the Moore neighborhood, inclusive.

    Returns
    -------
    set[Vertex]
        Selected coarse vertices.
    """
    coarse_vertices = set()
    visited = set()
    for vertex in graph.vertex_list:
        if vertex in visited:
            continue
        coarse_vertices.add(vertex)
        visited.update(get_moore_neighborhood(vertex, size))
    graph.set_coarse(coarse_vertices)
    return coarse_vertices

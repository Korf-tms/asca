import logging

from graph import OriginalGraph, Vertex
from graphutils import get_mis_set, get_moore_neighborhood, get_mis_set_ordered

logger = logging.getLogger(__name__)


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

    logger.info("Using MIS coarse selection")

    if size < 1:
        raise ValueError("size must be at least 1")
    mis = get_mis_set(graph.vertex_list, size)

    if not mis:
        raise ValueError("No coarse vertices selected")
    graph.set_coarse(mis)
    logger.info(f"Number of coarse vertices: {len(mis)}")
    return mis

def select_coarse_mis_min(graph: OriginalGraph, size: int = 1) -> set[Vertex]:
    """
    Select coarse vertices using a maximal independent set (MIS) with vertices sorted by their degree.

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

    logger.info("Using MIS min coarse selection")

    if size < 1:
        raise ValueError("size must be at least 1")
    mis = get_mis_set_ordered(list(sorted(graph.vertex_list, key=lambda v: len(v.adj), reverse=True)), size)
    if not mis:
        raise ValueError("No coarse vertices selected")
    graph.set_coarse(mis)
    logger.info(f"Number of coarse vertices: {len(mis)}")
    return mis

def select_coarse_mis_max(graph: OriginalGraph, size: int = 1) -> set[Vertex]:
    """
    Select coarse vertices using a maximal independent set (MIS) with vertices sorted by their degree.

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

    logger.info("Using MIS min coarse selection")

    if size < 1:
        raise ValueError("size must be at least 1")
    mis = get_mis_set_ordered(list(sorted(graph.vertex_list, key=lambda v: len(v.adj), reverse=False)), size)
    if not mis:
        raise ValueError("No coarse vertices selected")
    graph.set_coarse(mis)
    logger.info(f"Number of coarse vertices: {len(mis)}")
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

    logger.info("Using MIS coarse selection")

    if size < 1:
        raise ValueError("size must be at least 1")
    coarse_vertices = set()
    visited = set()
    for vertex in graph.vertex_list:
        if vertex in visited:
            continue
        coarse_vertices.add(vertex)
        visited.update(get_moore_neighborhood(vertex, size))
    if not coarse_vertices:
        raise ValueError("No coarse vertices selected")
    graph.set_coarse(coarse_vertices)
    logger.info(f"Number of coarse vertices: {len(coarse_vertices)}")
    return coarse_vertices

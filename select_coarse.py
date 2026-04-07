from graph import OriginalGraph, Vertex
from graphutils import get_mis_set, get_neighborhood_connectivity

def select_coarse_mis(graph: OriginalGraph, size: int = 1) -> set[Vertex]:
    mis = get_mis_set(graph.vertex_list, size)
    graph.set_coarse(mis)
    return mis

def select_coarse_moore(graph: OriginalGraph, size: int = 1) -> set[Vertex]:
    coarse_vertices = set()
    visited = set()
    for vertex in graph.vertex_list:
        if vertex in visited:
            continue
        coarse_vertices.add(vertex)
        visited.update(get_neighborhood_connectivity(vertex, size))
    graph.set_coarse(coarse_vertices)
    return coarse_vertices

from collections import defaultdict, deque

from graph import Vertex, Edge, OriginalGraph
from graphutils import get_moore_neighborhood, get_mis_set, get_neighborhood


def _get_moore_subgraph(vertices: set[Vertex] | list[Vertex], size: int):
    for vertex in vertices:
        degree = len(vertex.get_adj())
        if degree <= 4:
            yield get_moore_neighborhood(vertex, size)
        else:
            yield get_neighborhood(vertex, size=size)


def _get_depth_subgraph(vertices, max_depth):
    for vertex in vertices:
        yield get_neighborhood(vertex, size=max_depth)


def moore_neighborhood_around_coarse(graph: OriginalGraph, size: int = 2):
    for iterator, subgraph in enumerate(
        _get_moore_subgraph(graph.coarse_vertices, size)
    ):
        if len([vertex for vertex in subgraph if vertex in graph.coarse_vertices]) < 3:
            continue
        graph.add_subgraph(subgraph, f"SubGraph{iterator}")


def moore_neighborhood_all(graph: OriginalGraph, size: int = 2):
    for iterator, subgraph in enumerate(_get_moore_subgraph(graph.vertex_list, size)):
        if len([vertex for vertex in subgraph if vertex in graph.coarse_vertices]) < 3:
            continue
        graph.add_subgraph(subgraph, f"SubGraph{iterator}")


def create_subgraphs_depth(graph: OriginalGraph, size: int = 2):
    for iterator, subgraph in enumerate(
        _get_depth_subgraph(graph.coarse_vertices, size)
    ):
        graph.add_subgraph(subgraph, f"SubGraph{iterator}")


def create_subgraphs_macrostructures(
    graph: OriginalGraph,
    microstructure_size=2,
    subgraph_structure_connectivity=2,
    macrostructure_microstructure_inclusion_distance=1,
):
    subgraphs = dict()
    subgraph_structure_mapping = {
        vertex: Vertex(vertex.id) for vertex in graph.coarse_vertices
    }
    for vertex in graph.coarse_vertices:
        subgraphs[subgraph_structure_mapping[vertex]] = set(
            get_neighborhood(vertex, microstructure_size)
        )
        visited = set({vertex})
        depth = defaultdict(lambda: 1000)
        depth[vertex] = 0
        queue = deque([vertex])
        while queue:
            current = queue.popleft()
            if current in graph.coarse_vertices:
                subgraph_structure_mapping[vertex].adj.add(
                    Edge(
                        subgraph_structure_mapping[vertex],
                        subgraph_structure_mapping[current],
                        1,
                    )
                )
            for neighbor in current.get_adj():
                if depth[current] + 1 < depth[neighbor]:
                    depth[neighbor] = depth[current] + 1
                if (
                    depth[current] >= subgraph_structure_connectivity
                    or neighbor in visited
                ):
                    continue
                visited.add(neighbor)
                queue.append(neighbor)
    macrostructure_centers = get_mis_set(subgraph_structure_mapping.values(), 1)
    for i, vertex in enumerate(macrostructure_centers):
        macrostructure = set()
        macrostructure.update(subgraphs[vertex])
        for neighbour in get_neighborhood(
            vertex, size=macrostructure_microstructure_inclusion_distance
        ):
            macrostructure.update(subgraphs[neighbour])
        graph.add_subgraph(macrostructure, f"SubGraph{i}")

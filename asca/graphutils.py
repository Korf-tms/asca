from collections import deque

from .graph import Vertex


def get_neighborhood(vertex: Vertex, size: int = 1) -> set[Vertex]:
    """
    Find all vertices that have distance up to size from vertex.

    Parameters
    ----------
    vertex : Vertex
        Center vertex.
    size : int, default=1
        Maximal inclusive distance from center vertex.

    Returns
    -------
    set[Vertex]
        Set of vertices, this includes the center vertex.
    """
    selected_vertices = set({vertex})
    for _ in range(size):
        selected_vertices.update(*(v.get_adj() for v in selected_vertices))
    return selected_vertices


def get_moore_neighborhood(vertex: Vertex, size: int = 1) -> set[Vertex]:
    """
    Compute the Moore neighborhood of a vertex.

    The Moore neighborhood finds all vertices up to distance of size plus corners.
    A vertex is considered a corner if it is connected to at least two vertices in the current layer.

    Parameters
    ----------
    vertex : Vertex
        Center vertex.
    size : int, default=1
        Maximal distance from center vertex, not including the corners.

    Returns
    -------
    set[Vertex]
        Set of vertices in the Moore neighborhood.
    """
    last_layer = set([vertex])
    moore_neighborhood = set({vertex})
    for _ in range(size):
        # find all adjacents from the last layer
        new_layer = set()
        new_layer.update(*(v.get_adj() for v in last_layer))

        # find possible corners, which are all adjacents of new layer
        possible_corners = set()
        possible_corners.update(*(v.get_adj() for v in new_layer))
        possible_corners.difference_update(moore_neighborhood)

        # find corners, vertices that have at least 2 adjacents in new layer
        corners = set()
        for v in possible_corners:
            if len(new_layer.intersection(v.get_adj())) >= 2:
                corners.add(v)

        # add to moore neighborhood
        new_layer.update(corners)
        moore_neighborhood.update(new_layer)
        last_layer = new_layer
    return moore_neighborhood


def get_neighborhood_connectivity(
    vertex: Vertex, size: int = 1, connectivity: int = 1
) -> set[Vertex]:
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
            if (
                neighbor_depth == size
                and len(set(neighbor.get_adj()).intersection(visited)) <= connectivity
            ):
                continue
            if current_depth >= size or neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append(neighbor)
    return visited


def get_mis_set(vertex_list: list[Vertex] | set[Vertex], size: int = 1) -> set[Vertex]:
    """
    Compute a maximal independent set (MIS) using a greedy strategy.

    A maximal independent set is a set of vertices such that:
    - No two vertices are within the specified distance.
    - No additional vertex can be added without violating this condition.

    Produces the same independent set on the same graph.

    Parameters
    ----------
    vertex_list : list[Vertex] or set[Vertex]
        Collection of vertices to select from.
    size : int, default=1
        Minimal distance from other vertices in the mis set.
        A value of 1 corresponds to the standard MIS definition.

    Returns
    -------
    set[Vertex]
        A maximal independent set.
    """
    mis_set = set()
    remaining_vertices = set(vertex_list)
    while remaining_vertices:
        current = remaining_vertices.pop()
        mis_set.add(current)
        remaining_vertices.difference_update(get_neighborhood(current, size=size))
    return mis_set


def get_mis_set_ordered(vertex_list: list[Vertex], size: int = 1) -> set[Vertex]:
    """
    Compute a maximal independent set (MIS) using a greedy strategy.

    A maximal independent set is a set of vertices such that:
    - No two vertices are within the specified distance.
    - No additional vertex can be added without violating this condition.

    Produces the same independent set on the same graph.
    Order of the input matters.

    Parameters
    ----------
    vertex_list : list[Vertex] or set[Vertex]
        Collection of vertices to select from.
    size : int, default=1
        Minimal distance from other vertices in the mis set.
        A value of 1 corresponds to the standard MIS definition.

    Returns
    -------
    set[Vertex]
        A maximal independent set.
    """
    mis_set = set()
    remaining_vertices = vertex_list

    while remaining_vertices:
        current = remaining_vertices.pop(0)
        mis_set.add(current)
        neighborhood = set(get_neighborhood(current, size=size))
        remaining_vertices = [v for v in remaining_vertices if v not in neighborhood]

    return mis_set

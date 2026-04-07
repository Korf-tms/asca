from collections import deque

from graph import Vertex


def get_neighborhood(vertex: Vertex, size: int = 1) -> set[Vertex]:
    selected_vertices = set({vertex})
    for _ in range(size):
        selected_vertices.update(*(v.get_adj() for v in selected_vertices))
    return selected_vertices


def get_neighborhood_connectivity(vertex: Vertex, size: int = 1) -> set[Vertex]:
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
                and len(set(neighbor.get_adj()).intersection(visited)) <= 1
            ):
                continue
            if current_depth >= size or neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append(neighbor)
    return visited


def get_mis_set(vertex_list: list[Vertex] | set[Vertex], size: int = 1) -> set[Vertex]:
    mis_set = set()
    remaining_vertices = set(vertex_list)
    while remaining_vertices:
        current = remaining_vertices.pop()
        mis_set.add(current)
        remaining_vertices.difference_update(get_neighborhood(current, size=size))
    return mis_set

from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
from numpy import asarray, ones, float64

from graph import SubGraph


def schur_complement(sorted_adjacency_matrix, num_coarse):
    if num_coarse <= 0:
        raise ValueError("Number of coarse vertices must be greater than 0.")

    degrees = asarray(sorted_adjacency_matrix.sum(axis=1)).ravel()
    graph_laplacian = diags(degrees, format="csr") - sorted_adjacency_matrix
    l_11 = graph_laplacian[:num_coarse, :num_coarse]
    l_22 = graph_laplacian[num_coarse:, num_coarse:].tocsc()
    l_21 = graph_laplacian[num_coarse:, :num_coarse].tocsc()
    l_12 = graph_laplacian[:num_coarse, num_coarse:]

    return l_11 - l_12 @ spsolve(l_22, l_21)


def get_contribution(subgraph: SubGraph) -> csr_matrix:
    mapping = local_to_global_mapping(subgraph)
    schur_complement = local_schur_complement(subgraph)
    contribution = mapping @ schur_complement @ mapping.T
    return contribution


def local_to_global_mapping(subgraph: SubGraph) -> csr_matrix:
    coarse = subgraph.sorted_vertex_list[: subgraph.coarse_vertices_count]
    row_ind = []
    col_ind = []
    mapping = subgraph.parent.sorted_vertex_adj_matrix_mapping
    for iterator, vertex in enumerate(coarse):
        row_ind.append(mapping[vertex])
        col_ind.append(iterator)
    return csr_matrix(
        (ones(len(row_ind)), (row_ind, col_ind)),
        shape=(subgraph.parent.coarse_vertices_count, subgraph.coarse_vertices_count),
        dtype=float64,
    )


def local_schur_complement(subgraph: SubGraph) -> csr_matrix:
    adjacency_matrix = subgraph.to_adj_matrix(
        divide_edges=True, sorting=subgraph.parent.vertex_sort
    )

    local_schur = schur_complement(adjacency_matrix, subgraph.coarse_vertices_count)
    return csr_matrix(local_schur, dtype=float64)

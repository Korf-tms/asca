from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
from numpy import asarray, ones, float64

from .graph import SubGraph


def schur_complement_blocks(graph_laplacian, num_coarse):
    """
    Split graph laplacian into Schur complement blocks.

    The graph Laplacian is assumed to be ordered so that coarse vertices
    come first.

    Parameters
    ----------
    graph_laplacian : csr_matrix
        Graph laplacian with coarse vertices first.
    num_coarse : int
        Number of coarse vertices.

    Returns
    -------
    (l_11, 1_22, l_21, l_12) : tuple[csr_matrix, csr_matrix, csr_matrix, csr_matrix]
        Blocks of the Schur complement, where l_11 is coarse to coarse and l_22 fine to fine.
    """
    l_11 = graph_laplacian[:num_coarse, :num_coarse]
    l_22 = graph_laplacian[num_coarse:, num_coarse:]
    l_21 = graph_laplacian[num_coarse:, :num_coarse]
    l_12 = graph_laplacian[:num_coarse, num_coarse:]
    return l_11, l_22, l_21, l_12

def schur_complement(sorted_adjacency_matrix, num_coarse):
    """
    Compute the Schur complement of a graph Laplacian.

    The adjacency matrix is assumed to be ordered so that coarse vertices
    come first.

    Parameters
    ----------
    sorted_adjacency_matrix : csr_matrix
        Adjacency matrix with coarse vertices first.
    num_coarse : int
        Number of coarse vertices.

    Returns
    -------
    csr_matrix
        Schur complement matrix for the coarse vertices.
    """
    if num_coarse <= 0:
        raise ValueError("Number of coarse vertices must be greater than 0.")

    degrees = asarray(sorted_adjacency_matrix.sum(axis=1)).ravel()
    graph_laplacian = diags(degrees, format="csr") - sorted_adjacency_matrix

    l_11, l_22, l_21, l_12 = schur_complement_blocks(graph_laplacian, num_coarse)

    l_22 = l_22.tocsc()
    l_21 = l_21.tocsc()

    return l_11 - l_12 @ spsolve(l_22, l_21)


def get_contribution(subgraph: SubGraph) -> csr_matrix:
    """
    Compute the global contribution of a subgraph.

    Parameters
    ----------
    subgraph : SubGraph

    Returns
    -------
    csr_matrix
        Contribution matrix in global coordinates.
    """
    mapping = local_to_global_mapping(subgraph)
    schur_complement = local_schur_complement(subgraph)
    contribution = mapping @ schur_complement @ mapping.T
    return contribution


def local_to_global_mapping(subgraph: SubGraph) -> csr_matrix:
    """
    Build a mapping from local coarse vertices to global indices.

    Parameters
    ----------
    subgraph : SubGraph

    Returns
    -------
    csr_matrix
        Sparse matrix mapping local coarse indices to global indices.
    """
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
    """
    Compute the Schur complement for a subgraph.

    Parameters
    ----------
    subgraph : SubGraph

    Returns
    -------
    csr_matrix
        Schur complement restricted to the subgraph's coarse vertices.
    """
    adjacency_matrix = subgraph.to_adj_matrix(
        divide_edges=True, sorting=subgraph.parent.vertex_sort
    )

    local_schur = schur_complement(adjacency_matrix, subgraph.coarse_vertices_count)
    return csr_matrix(local_schur, dtype=float64)

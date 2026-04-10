from scipy.sparse import coo_matrix, issparse
import pandas as pd
import h5py
import scipy.io as spio
import pathlib as pl

from graph import Edge, Graph, Vertex


def from_file(path: str | pl.Path, cls=Graph):
    actual_path = pl.Path(path)
    if not actual_path.exists():
        raise ValueError("File does not exist.")

    if actual_path.suffix == ".csv":
        return from_csv(path, cls)
    if actual_path.suffix == ".hdf5":
        return from_hdf5(path, cls)
    if actual_path.suffix == ".mat":
        return from_mat(path, cls)
    if actual_path.suffix == ".mtx":
        return from_mtx(path, cls)


def from_coo(rows=None, cols=None, values=None, coo_mat: coo_matrix = None, cls=Graph):
    """
    Create a graph from COO (coordinate) sparse matrix data.

    The COO format stores non-zero entries of a sparse matrix as three arrays
    such that:

        A[rows[i], cols[i]] = values[i]

    Exactly one of the following input forms must be provided:

    - rows, cols, and values
    - coo_mat

    Parameters
    ----------
    rows : array-like of int, optional
        Row indices of non-zero entries.
    cols : array-like of int, optional
        Column indices of non-zero entries.
    values : array-like, optional
        Values corresponding to each (row, col) pair.
    coo_mat : scipy.sparse.coo_matrix, optional
        A SciPy COO sparse matrix whose row indices, column indices, and data
        are used to construct the graph.
    cls : type, default=Graph
        Graph class to instantiate.

    Returns
    -------
    Graph
        An instance of cls constructed from the COO representation.
    """
    provided_triplet = rows is not None or cols is not None or values is not None
    provided_all_triplet = rows is not None and cols is not None and values is not None
    provided_coo = coo_mat is not None

    if provided_coo and provided_triplet:
        raise ValueError("Provide either coo_mat or (rows, cols, values), not both.")

    if not provided_coo and not provided_all_triplet:
        raise ValueError(
            "You must provide either coo_mat or all of rows, cols, and values."
        )

    if provided_coo:
        rows = coo_mat.row
        cols = coo_mat.col
        values = coo_mat.data

    if not (len(rows) == len(cols) == len(values)):
        raise ValueError("rows, cols, and values must have the same length.")

    if len(rows) == 0:
        return cls([])

    n = int(max(max(rows), max(cols)) + 1)
    vertex_dictionary = {i: Vertex(i) for i in range(n)}

    for row, col, val in zip(rows, cols, values):
        if row >= col:
            continue
        vertex_row = vertex_dictionary[int(row)]
        vertex_col = vertex_dictionary[int(col)]
        edge = Edge(vertex_row, vertex_col, val)
        vertex_row.adj.add(edge)
        vertex_col.adj.add(edge)
    return cls(set(vertex_dictionary.values()))


def from_csv(path, cls=Graph):
    """
    Load a graph from a CSV file containing COO-format data.

    The CSV file must contain columns:
        - row
        - col
        - val

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    Graph
        Graph constructed from the CSV data.
    """
    dataframe = pd.read_csv(path)

    return from_coo(dataframe["row"], dataframe["col"], dataframe["val"], cls=cls)


def from_hdf5(path, cls=Graph):
    """
    Load a graph from an HDF5 file.

    Supported formats inside the file:
        - coo_matrix group with datasets row, col, val
        - adj_matrix dataset (dense or sparse)

    Parameters
    ----------
    path : str
        Path to the HDF5 file.

    Returns
    -------
    Graph
        Graph constructed from the HDF5 data.
    """
    with h5py.File(path, "r") as file:
        if "coo_matrix" in file:
            group = file["coo_matrix"]

            return from_coo(group["row"][:], group["col"][:], group["val"][:], cls=cls)

        elif "adj_matrix" in file:
            # Convert adjacency matrix to COO format
            adj = coo_matrix(file["adj_matrix"])
            return from_coo(coo_mat=adj, cls=cls)

        else:
            raise ValueError(
                f"HDF5 file {path} does not contain coo_matrix or adj_matrix."
            )


def from_mat(path, cls=Graph):
    """
    Load a graph from a MATLAB (.mat) file.

    Expected structure:
        mat["Problem"][0][0][1] or mat["Problem"][0][0][2]
    should contain a sparse adjacency matrix.

    Parameters
    ----------
    path : str
        Path to the .mat file.

    Returns
    -------
    Graph
        Graph constructed from the MATLAB data.
    """
    mat = spio.loadmat(path)

    if "Problem" not in mat:
        raise ValueError(f"MAT file {path} does not contain Problem key.")

    adj = mat["Problem"][0][0][1]

    if not hasattr(adj, "indptr"):
        adj = mat["Problem"][0][0][2]

    coo_adj = adj.tocoo()

    return from_coo(coo_mat=coo_adj, cls=cls)


def from_mtx(path, cls=Graph):
    """
    Load a graph from a Matrix Market (.mtx) file.

    Supports both sparse and dense matrices.

    Parameters
    ----------
    path : str
        Path to the .mtx file.

    Returns
    -------
    Graph
        Graph constructed from the matrix data.
    """
    adj_mat = spio.mmread(path)

    if issparse(adj_mat):
        coo_adj_mat = (
            adj_mat.tocoo() if not isinstance(adj_mat, coo_matrix) else adj_mat
        )
    else:
        coo_adj_mat = coo_matrix(adj_mat)

    return from_coo(coo_mat=coo_adj_mat, cls=cls)

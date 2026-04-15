from scipy.sparse import coo_matrix, issparse
import pandas as pd
import h5py
import scipy.io as spio
import pathlib as pl

from graph import Edge, Graph, Vertex


def from_file(path: str | pl.Path, cls=Graph):
    """
    Load a graph from a supported file format.

    Supported formats
    -----------------
    - .csv: COO triplet data with columns row, col, and val
    - .hdf5: HDF5 file containing either coo_matrix or adj_matrix
    - .mat: MATLAB file containing a sparse adjacency matrix under mat["Problem"]
    - .mtx: Matrix Market file containing a dense or sparse matrix

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the input file.
    cls : type, default=Graph
        Graph class to instantiate, must be child of Graph class.

    Returns
    -------
    Graph
        An instance of cls loaded from the file contents.
    """
    actual_path = pl.Path(path)
    if not actual_path.exists():
        raise ValueError("File does not exist.")

    if actual_path.suffix == ".csv":
        return from_csv(path, cls)
    if actual_path.suffix == ".hdf5":
        return from_hdf5(path, cls)
    if actual_path.suffix == ".mat":
        return from_mat(path, cls)
    if actual_path.suffix == ".gz":
        return from_mtx(path, cls)
    raise ValueError("Unsupported filetype.")


def from_coo(rows=None, cols=None, values=None, coo_mat: coo_matrix = None, cls=Graph):
    """
    Create a graph from COO sparse matrix data.

    The COO (coordinate) format stores non-zero entries of a sparse matrix
    using three arrays so that:

        A[rows[i], cols[i]] = values[i]

    Exactly one of the following input forms must be provided:

    - rows, cols, and values
    - coo_mat

    Parameters
    ----------
    rows : array-like of int, optional
        Row indices of matrix entries.
    cols : array-like of int, optional
        Column indices of matrix entries.
    values : array-like, optional
        Values of matrix entries.
    coo_mat : scipy.sparse.coo_matrix, optional
        SciPy COO sparse matrix
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
        if row == col:
            continue

        u = min(int(row), int(col))
        v = max(int(row), int(col))

        vertex_row = vertex_dictionary[u]
        vertex_col = vertex_dictionary[v]
        edge = Edge(vertex_row, vertex_col, val)
        vertex_row.adj.add(edge)
        vertex_col.adj.add(edge)
    return cls(set(vertex_dictionary.values()))


def from_csv(path, cls=Graph):
    """
    Load a graph from a CSV file containing COO-format data.

    The CSV file must contain the following columns:

    - row: row indices
    - col: column indices
    - val: entry values

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the CSV file.
    cls : type, default=Graph
        Graph class to instantiate.

    Returns
    -------
    Graph
        An instance of cls constructed from the CSV data.
    """
    dataframe = pd.read_csv(path)

    return from_coo(dataframe["row"], dataframe["col"], dataframe["val"], cls=cls)


def from_hdf5(path, cls=Graph):
    """
    Load a graph from an HDF5 file.

    Supported hdf5 groups are:

    - coo_matrix groups containing coo sparse matrix stored in row, col, and val datasets
    - adj_matrix dataset containing the dense adjacency matrix

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the HDF5 file.
    cls : type, default=Graph
        Graph class to instantiate.

    Returns
    -------
    Graph
        An instance of cls constructed from the HDF5 data.
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
    Load a graph from a MATLAB .mat file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the MATLAB file.
    cls : type, default=Graph
        Graph class to instantiate.

    Returns
    -------
    Graph
        An instance of cls constructed from the MATLAB data.
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
    Load a graph from a Matrix Market .mtx file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the Matrix Market file.
    cls : type, default=Graph
        Graph class to instantiate.

    Returns
    -------
    Graph
        An instance of cls constructed from the matrix data.
    """
    adj_mat = spio.mmread(path)

    if issparse(adj_mat):
        coo_adj_mat = (
            adj_mat.tocoo() if not isinstance(adj_mat, coo_matrix) else adj_mat
        )
    else:
        coo_adj_mat = coo_matrix(adj_mat)

    return from_coo(coo_mat=coo_adj_mat, cls=cls)

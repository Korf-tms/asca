import pathlib as pl
import h5py
from scipy.sparse import csr_matrix
import numpy as np


def create_folder(folder_path: str | pl.Path):
    folder = None
    if isinstance(folder_path, pl.Path):
        folder = folder_path
    else:
        folder = pl.Path(folder_path)

    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)


def write_csr_matrix(group: h5py.Group | h5py.File, matrix: csr_matrix):
    group.create_dataset("data", data=matrix.data)
    group.create_dataset("indices", data=matrix.indices)
    group.create_dataset("indptr", data=matrix.indptr)
    group.create_dataset("shape", data=matrix.shape)


def read_csr_matrix(group: h5py.Group | h5py.File) -> csr_matrix:
    data = group["data"][:]
    indices = group["indices"][:]
    indptr = group["indptr"][:]
    shape = tuple(group["shape"][:])

    return csr_matrix((data, indices, indptr), shape=shape, dtype=np.float64)


def read_int(group: h5py.Group | h5py.File, key: str):
    return int(group[key][()])


def write_data(filename: str, group: str, data: dict):
    with h5py.File(filename, mode="a") as file:
        group = file.require_group(group)

        for key, value in data.items():
            if key in group:
                del group[key]

            if np.isscalar(value):
                group.create_dataset(key, data=value)
            else:
                group.create_dataset(key, data=np.asarray(value))

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


def get_unique_path(
    base_name, output_file=None, data_folder=None, name=None, suffix="hdf5"
):
    if output_file is not None:
        output_path = pl.Path(output_file)
        if not output_path.parent.exists() and data_folder is not None:
            output_path = pl.Path(data_folder) / output_path
    else:
        output_name = f"{base_name}_{name}"
        output_path = pl.Path(data_folder) / f"{output_name}.{suffix}"

    file_num = 0
    unique_path = output_path

    while unique_path.exists():
        unique_path = output_path.with_name(
            f"{output_path.stem}_{file_num}{output_path.suffix}"
        )
        file_num += 1

    return unique_path


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

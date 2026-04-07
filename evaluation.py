from scipy.sparse import csr_matrix, eye, diags
from scipy.sparse.linalg import spsolve, eigsh, cgs, LinearOperator

import pathlib as pl
import numpy as np
import h5py
import utils
from asca import DATA_FOLDER


class Evaluator:
    def __init__(self, input_file: str, output_file: str = None):
        self.input_file = pl.Path(input_file)
        self.output_file = utils.get_unique_path(
            self.input_file.stem,
            output_file=output_file,
            data_folder=DATA_FOLDER,
            name="evaluation",
            suffix="hdf5",
        )
        if not self.input_file.exists():
            raise ValueError("File doesnt exist.")

    def _get_iterations(self):
        with h5py.File(self.input_file, mode="r") as file:
            iterations = []
            for key in file.keys():
                if key.startswith("iteration"):
                    iterations.append(int(key.replace("iteration", "")))
            iterations = sorted(i for i in iterations if i != 0)
            return iterations

    def _read_matrix(self, key: str) -> csr_matrix:
        with h5py.File(self.input_file, mode="r") as file:
            if key not in file:
                raise ValueError("Key not found.")

            matrix = file[key]
            data = matrix["data"][:]
            indices = matrix["indices"][:]
            indptr = matrix["indptr"][:]
            shape = tuple(matrix["shape"][:])

            return csr_matrix((data, indices, indptr), shape=shape, dtype=np.float64)

    def _read_iteration(self, iteration: int) -> csr_matrix:
        return self._read_matrix(f"iteration{iteration}")

    def _get_schur_complement(self) -> csr_matrix:
        adj_mat = self._read_matrix("adj_matrix")
        coarse_count = 0

        with h5py.File(self.input_file, mode="r") as file:
            coarse_count = int(file["adj_matrix/coarse_count"][()])

        degrees = np.asarray(adj_mat.sum(axis=1)).ravel()
        graph_laplacian = diags(degrees, format="csr") - adj_mat

        l_11 = graph_laplacian[:coarse_count, :coarse_count]
        l_22 = graph_laplacian[coarse_count:, coarse_count:].tocsc()
        l_21 = graph_laplacian[coarse_count:, :coarse_count].tocsc()
        l_12 = graph_laplacian[:coarse_count, coarse_count:]

        return csr_matrix(l_11 - l_12 @ spsolve(l_22, l_21))

    def _write_iteration_data(self, iteration: int, data: dict):
        with h5py.File(self.output_file, mode="a") as file:
            group = file.require_group(f"iteration{iteration}")

            for key, value in data.items():
                if key in group:
                    del group[key]

                if np.isscalar(value):
                    group.create_dataset(key, data=value)
                else:
                    group.create_dataset(key, data=np.asarray(value))

    def cgs_evaluation(self, iteration: list[int] = None):
        iterations = iteration if iteration else self._get_iterations()

        for current_iteration in iterations:
            last_matrix = (
                self._read_iteration(current_iteration - 1)
                if current_iteration != 1
                else self._get_schur_complement()
            )
            current_matrix = self._read_iteration(current_iteration)

            last_matrix = last_matrix + eye(last_matrix.shape[0], format="csr") * 1e-5
            current_matrix = (
                current_matrix + eye(current_matrix.shape[0], format="csr") * 1e-5
            )

            current_matrix_inv = LinearOperator(
                shape=current_matrix.shape,
                matvec=lambda x: spsolve(current_matrix, x),
                dtype=np.float64,
            )

            b = np.random.rand(current_matrix.shape[0])
            x, info = cgs(
                A=last_matrix,
                M=current_matrix_inv,
                b=b,
            )

            self._write_iteration_data(
                current_iteration,
                {
                    "vertices": last_matrix.shape[0],
                    "coarse_vertices": current_matrix.shape[0],
                    "cgs_iterations": info,
                },
            )

    def eigsh_evaluation(self, iteration: list[int] = []):
        iterations = iteration if iteration else self._get_iterations()

        for current_iteration in iterations:
            last_matrix = (
                self._read_iteration(current_iteration - 1)
                if current_iteration != 1
                else self._get_schur_complement()
            )
            current_matrix = self._read_iteration(current_iteration)

            eigenvalues, eigenvectors = eigsh(A=last_matrix, M=current_matrix)

            self._write_iteration_data(
                current_iteration,
                {
                    "eigenvalues": eigenvalues,
                    "eigenvectors": eigenvectors,
                    "vertices": last_matrix.shape[0],
                    "coarse_vertices": current_matrix.shape[0],
                },
            )

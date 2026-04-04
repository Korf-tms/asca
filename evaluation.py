from scipy.sparse import csr_matrix, eye, diags
from scipy.sparse.linalg import spsolve, eigsh, cgs, LinearOperator, eigs

import pathlib as pl
import numpy as np
import h5py
import matplotlib.pyplot as plt

class Evaluator:
    def __init__(self, path : str):
        self.path = pl.Path(path)
        if not self.path.exists():
            raise ValueError("File doesnt exist.")

    def _get_iteratios(self):
        with h5py.File(self.path, mode="r") as file:
            iterations = []
            for key in file.keys():
                if key.startswith("iteration"):
                    iterations.append(int(key.replace("iteration", "")))
            iterations = sorted(i for i in iterations if i != 0)
            return iterations

    def _read_matrix(self, key):
        with h5py.File(self.path, mode="r") as file:
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

    def _get_schur_complement(self):
        adj_mat = self._read_matrix(f"adj_matrix")
        coarse_count = 0
        with h5py.File(self.path, mode="r") as file:
            coarse_count = int(file["adj_matrix/coarse_count"][()])
        
        degrees = np.asarray(adj_mat.sum(axis=1)).ravel()
        graph_laplacian = diags(degrees, format="csr") - adj_mat
        l_11 = graph_laplacian[:coarse_count, :coarse_count]
        l_22 = graph_laplacian[coarse_count:, coarse_count:].tocsc()
        l_21 = graph_laplacian[coarse_count:, :coarse_count].tocsc()
        l_12 = graph_laplacian[:coarse_count, coarse_count:]

        return csr_matrix(l_11 - l_12 @ spsolve(l_22, l_21))

    def cgs_evaluation(self, iteration : list[int] = []):
        iterations = iteration if iteration else self._get_iteratios()
        for current_iteration in iterations:
            last_matrix = self._read_iteration(current_iteration - 1) if current_iteration != 1 else self._get_schur_complement()
            current_matrix = self._read_iteration(current_iteration)

            last_matrix = last_matrix + eye(last_matrix.shape[0]) * 1e-5
            current_matrix = current_matrix + eye(current_matrix.shape[0]) * 1e-5

            current_matrix_inv = LinearOperator(
                shape=current_matrix.shape,
                matvec=lambda x: spsolve(current_matrix, x),
                dtype=np.float64
            )

            b = np.random.rand(current_matrix.shape[0])
            x, info = cgs(
                A=last_matrix, 
                M=current_matrix_inv,
                b=b,
                )
            
            return info
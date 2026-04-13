from scipy.sparse import eye
from scipy.sparse.linalg import eigsh, cg, inv
import pathlib as pl
import numpy as np
import h5py
import utils

from asca import DATA_FOLDER
from schur_complement import schur_complement


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

        self.schur_dict = dict()
        self.approximation_dict = dict()

        utils.create_folder(DATA_FOLDER)

        if not self.input_file.exists():
            raise ValueError("File doesnt exist.")

    def _get_iterations(self):
        with h5py.File(self.input_file, mode="r") as file:
            iterations = []
            for key in file.keys():
                if key.startswith("iteration"):
                    iterations.append(int(key.replace("iteration", "")))
            iterations = sorted(i for i in iterations if i != max(iterations))
            return iterations

    def _read_iteration(self, iteration: int):
        with h5py.File(self.input_file, mode="r") as file:
            iteration_group = file[f"iteration{iteration}"]
            mat = utils.read_csr_matrix(iteration_group["adj_matrix"])

            coarse = 0
            if "coarse_count" in iteration_group.keys():
                coarse = int(iteration_group["coarse_count"][()])
        return mat, coarse

    def _get_matrices(self, iteration: int):
        approximation, schur, vertices, vertices_coarse = 0, 0, 0, 0

        if iteration in self.schur_dict:
            schur = self.schur_dict[iteration]
        else:
            adj_mat, vertices_coarse = self._read_iteration(iteration)
            schur = schur_complement(adj_mat, vertices_coarse)

        if iteration in self.approximation_dict:
            approximation = self.approximation_dict
        else:
            approximation, _ = self._read_iteration(iteration + 1)

        vertices = adj_mat.shape[0]
        vertices_coarse = schur.shape[0]

        approximation += eye(approximation.shape[0], format="csr") * 1e-5
        schur += eye(schur.shape[0], format="csr") * 1e-5

        return schur, approximation, vertices, vertices_coarse

    def cg_evaluation(self, iteration: list[int] = None):
        iterations = iteration if iteration else self._get_iterations()

        for current_iteration in iterations:
            approximation_matrix, schur_matrix, vertex_count, coarse_count = (
                self._get_matrices(current_iteration)
            )

            ones = np.ones(approximation_matrix.shape[0])

            x_exact = np.random.rand(approximation_matrix.shape[0])
            x_exact = x_exact - np.dot(ones, x_exact) * ones / np.linalg.norm(ones)

            rhs = schur_matrix @ x_exact

            schur_matrix_inv = inv(schur_matrix.tocsc())

            iteration_count = 0
            error_history = []

            def cg_callback(x_current):
                nonlocal iteration_count, error_history
                iteration_count += 1
                error_history.append(np.linalg.norm(x_exact - x_current))

            _, info = cg(
                A=approximation_matrix, M=schur_matrix_inv, b=rhs, callback=cg_callback
            )

            utils.write_data(
                self.output_file,
                f"iteration{current_iteration}/",
                {
                    "vertex_count": vertex_count,
                    "coarse_count": coarse_count,
                    "iteration_count": iteration_count,
                    "info": info,
                    "error_history": error_history,
                },
            )

    def eigsh_evaluation(self, iteration: list[int] = []):
        iterations = iteration if iteration else self._get_iterations()

        for current_iteration in iterations:
            approximation_matrix, schur_matrix, _, _ = self._get_matrices(
                current_iteration
            )

            eigenvalues, eigenvectors = eigsh(A=schur_matrix, M=approximation_matrix)

            utils.write_data(
                self.output_file,
                f"iteration{current_iteration}/",
                {
                    "eigenvalues": eigenvalues,
                    "eigenvectors": eigenvectors,
                },
            )

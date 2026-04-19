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

        utils.create_folder(DATA_FOLDER)

        if not self.input_file.exists():
            raise ValueError("File doesnt exist.")

    def _get_iterations(self):
        with h5py.File(self.input_file, mode="r") as file:
            iterations = []
            for key in file.keys():
                if key.startswith("iteration"):
                    iterations.append(int(key.replace("iteration", "")))
            return list(sorted(iterations))

    def _read_iteration(self, iteration: int):
        with h5py.File(self.input_file, mode="r") as file:
            iteration_group = file[f"iteration{iteration}"]
            adj_mat = utils.read_csr_matrix(iteration_group["adj_matrix"])
            approximation = utils.read_csr_matrix(iteration_group["approximation"])
            coarse = int(iteration_group["coarse_count"][()])
            subgraph_mean = int(iteration_group["subgraph_size"][()])
        return adj_mat, approximation, coarse, subgraph_mean

    def _get_matrices(self, iteration: int):
        adjacency_matrix, approximation, coarse_count, subgraph_mean = (
            self._read_iteration(iteration)
        )

        if iteration in self.schur_dict:
            schur = self.schur_dict[iteration]
        else:
            schur = schur_complement(adjacency_matrix, coarse_count)
            schur = schur + eye(schur.shape[0], format="csr") * 1e-5
            self.schur_dict[iteration] = schur

        approximation = approximation + eye(approximation.shape[0], format="csr") * 1e-5

        vertices = adjacency_matrix.shape[0]
        vertices_coarse = coarse_count

        return schur, approximation, vertices, vertices_coarse, subgraph_mean

    def cg_evaluation(self, iteration: list[int] = None):
        iterations = iteration if iteration else self._get_iterations()

        for current_iteration in iterations:
            (
                schur_matrix,
                approximation_matrix,
                vertex_count,
                coarse_count,
                mean_subgraph,
            ) = self._get_matrices(current_iteration)

            ones = np.ones(approximation_matrix.shape[0])

            x_exact = np.random.rand(approximation_matrix.shape[0])
            x_exact = x_exact - np.dot(ones, x_exact) * ones / np.linalg.norm(ones)

            rhs = schur_matrix @ x_exact

            approximation_matrix_inv = inv(approximation_matrix.tocsc())

            iteration_count = 0
            error_history = []

            def cg_callback(x_current):
                nonlocal iteration_count, error_history
                iteration_count += 1
                error_history.append(np.linalg.norm(x_exact - x_current))

            x, info = cg(
                A=schur_matrix, M=approximation_matrix_inv, b=rhs, callback=cg_callback
            )

            error_history.append(np.linalg.norm(x_exact - x))

            utils.write_data(
                self.output_file,
                f"iteration{current_iteration}/",
                {
                    "vertex_count": vertex_count,
                    "coarse_count": coarse_count,
                    "iteration_count": iteration_count,
                    "info": info,
                    "error_history": error_history,
                    "mean_subgraph": mean_subgraph,
                },
            )

    def eigsh_evaluation(self, iteration: list[int] = None):
        iterations = iteration if iteration else self._get_iterations()

        for current_iteration in iterations:
            schur_matrix, approximation_matrix, _, _, _ = self._get_matrices(
                current_iteration
            )

            eigenvalues, eigenvectors = eigsh(A=schur_matrix, M=approximation_matrix)

            condition_number = eigenvalues.max() / eigenvalues.min()
            eigenvalues_mean = np.mean(eigenvalues)

            utils.write_data(
                self.output_file,
                f"iteration{current_iteration}/",
                {
                    "eigenvalues": eigenvalues,
                    "eigenvectors": eigenvectors,
                    "condition_number": condition_number,
                    "eigenvalues_mean": eigenvalues_mean,
                },
            )

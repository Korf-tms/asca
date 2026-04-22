import logging
import pathlib as pl
import time
from dataclasses import dataclass

import h5py
import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import LinearOperator, cg, eigsh, factorized

from . import utils
from .schur_complement import schur_complement

logger = logging.getLogger(__name__)

SCHUR_CACHE_GROUP = "schur_complement"
NORMALIZATION_EPSILON = 1e-5


@dataclass
class Iteration:
    approximation_matrix: csr_matrix | None = None
    schur_complement_matrix: csr_matrix | None = None
    vertex_count: int | None = None
    coarse_count: int | None = None
    subgraph_size_mean: int | None = None
    iteration: int | None = None


class EvaluatorConfig:
    def __init__(
        self,
        input_files: str | pl.Path | list[str] | list[pl.Path],
        evaluate_cg: bool = True,
        evaluate_eigv: bool = True,
        output_directory: str | pl.Path | None = None,
    ):
        self.input_files = (
            [pl.Path(input_files)]
            if isinstance(input_files, (str, pl.Path))
            else [pl.Path(file) for file in input_files]
        )
        self.evaluate_cg = evaluate_cg
        self.evaluate_eigv = evaluate_eigv
        self.output_directory = (
            pl.Path(output_directory)
            if output_directory is not None
            else pl.Path("evaluation/")
        )


def run_evaluation(config: EvaluatorConfig):

    for input_file in config.input_files:

        if not input_file.exists():
            raise ValueError(f"Input file does not exist: {input_file}")

        iterations = _get_iterations(input_file)

        output_file = _get_output_file(config, input_file)
        utils.create_folder(output_file.parent)

        logger.info("Evaluation input: %s", input_file)
        logger.info("Evaluation output: %s", output_file)

        selected_iterations = (
            iterations if iterations is not None else _get_iterations(input_file)
        )
        logger.info("Found %s iteration(s) to evaluate", len(selected_iterations))

        for iteration in selected_iterations:
            logger.info("Iteration %s: starting evaluation", iteration)

            iteration_data = _get_matrices(
                input_file,
                iteration,
            )

            output = dict()

            if config.evaluate_cg:
                output.update(cg_evaluation(iteration_data))
            if config.evaluate_eigv:
                output.update(eigsh_evaluation(iteration_data))

            output["vertex_count"] = iteration_data.vertex_count
            output["coarse_count"] = iteration_data.coarse_count
            output["mean_subgraph"] = iteration_data.subgraph_size_mean

            logger.info("Iteration %s: writing evaluation output", iteration)
            utils.write_data(output_file, f"iteration{iteration}", output)

    logger.info("Evaluation finished")


def _get_iterations(input_file: pl.Path) -> list[int]:
    with h5py.File(input_file, mode="r") as file:
        iterations = []
        for key in file.keys():
            if key.startswith("iteration"):
                iterations.append(int(key.replace("iteration", "")))

    return list(sorted(iterations))


def _get_output_file(config: EvaluatorConfig, input_file: pl.Path) -> pl.Path:
    output_name = f"{input_file.stem}_evaluation.hdf5"
    if config.output_directory is not None:
        return (
            config.output_directory / output_name
            if config.output_directory is not None
            else output_name
        )

    return input_file.with_name(output_name)


def _normalize_matrix(matrix: csr_matrix) -> csr_matrix:
    return matrix + eye(matrix.shape[0], format="csr") * NORMALIZATION_EPSILON


def _get_matrices(
    input_file: pl.Path,
    iteration: int,
) -> Iteration:
    iteration_data = Iteration(iteration=iteration)

    logger.info("Iteration %s: reading matrices", iteration)

    with h5py.File(input_file, mode="r") as file:
        iteration_group = file[f"iteration{iteration}"]

        adjacency_matrix = utils.read_csr_matrix(iteration_group["adj_matrix"])
        iteration_data.vertex_count = adjacency_matrix.shape[0]

        iteration_data.approximation_matrix = utils.read_csr_matrix(
            iteration_group["approximation"]
        )
        iteration_data.coarse_count = utils.read_int(iteration_group, "coarse_count")
        iteration_data.subgraph_size_mean = utils.read_int(
            iteration_group, "subgraph_size"
        )

    iteration_data.approximation_matrix = _normalize_matrix(
        iteration_data.approximation_matrix
    )

    cache_key = _schur_cache_key(adjacency_matrix, iteration_data.coarse_count)

    schur = None

    for file in pl.Path("schur_cache").glob("*.hdf5"):
        if cache_key in file.stem:
            with h5py.File(file) as schur_cache:
                schur = utils.read_csr_matrix(schur_cache)
                continue

    if schur is None:
        logger.info("Iteration %s: computing exact Schur complement", iteration)
        start_time = time.perf_counter()
        schur = schur_complement(adjacency_matrix, iteration_data.coarse_count)

        logger.info(
            "Iteration %s: exact Schur complement computed in %.3fs",
            iteration,
            time.perf_counter() - start_time,
        )
        utils.create_folder("schur_cache")
        with h5py.File(f"schur_cache/{cache_key}.hdf5", mode="w") as file:
            utils.write_csr_matrix(file, schur)

    iteration_data.schur_complement_matrix = _normalize_matrix(schur)

    return iteration_data


def _schur_cache_key(
    adjacency_matrix: csr_matrix,
    coarse_count: int,
) -> tuple:
    return str(hash((adjacency_matrix.shape, adjacency_matrix.nnz, coarse_count)))


def cg_evaluation(iteration_data: Iteration):
    logger.info("Iteration %s: starting CG evaluation", iteration_data.iteration)

    ones = np.ones(iteration_data.approximation_matrix.shape[0])

    x_exact = np.random.rand(iteration_data.approximation_matrix.shape[0])
    x_exact = x_exact - (ones @ x_exact) / (ones @ ones) * ones

    rhs = iteration_data.schur_complement_matrix @ x_exact

    logger.info(
        "Iteration %s: factorizing approximation matrix", iteration_data.iteration
    )
    solve_m = factorized(iteration_data.approximation_matrix.tocsc())

    linear_op = LinearOperator(
        shape=iteration_data.approximation_matrix.shape,
        matvec=solve_m,
        dtype=np.float64,
    )

    iteration_count = 0
    error_history = []

    def cg_callback(x_current):
        nonlocal iteration_count, error_history
        iteration_count += 1
        error_history.append(np.linalg.norm(x_exact - x_current))

    _, info = cg(
        A=iteration_data.schur_complement_matrix,
        M=linear_op,
        b=rhs,
        callback=cg_callback,
    )

    return {
        "iteration_count": iteration_count,
        "info": info,
        "error_history": error_history,
    }


def eigsh_evaluation(iteration_data: Iteration):
    logger.info(
        "Iteration %s: starting eigen evaluation",
        iteration_data.iteration,
    )

    k = min(6, iteration_data.schur_complement_matrix.shape[0] - 1)
    start_time = time.perf_counter()
    largest_eigenvalues = eigsh(
        A=iteration_data.schur_complement_matrix,
        M=iteration_data.approximation_matrix,
        k=k,
        tol=1e-5,
        which="LA",
        return_eigenvectors=False
    )

    smallest_eigenvalues = eigsh(
        A=iteration_data.schur_complement_matrix,
        M=iteration_data.approximation_matrix,
        k=k,
        tol=1e-5,
        which="SA",
        return_eigenvectors=False
    )

    eigsh_time = time.perf_counter() - start_time
    
    condition_number = max(largest_eigenvalues) / min(smallest_eigenvalues)

    logger.info(
        "Iteration %s: eigen solve took %fs",
        iteration_data.iteration,
        eigsh_time,
    )

    return {
        "largest_eigenvalues": largest_eigenvalues,
        "smallest_eigenvalues" : smallest_eigenvalues,
        "condition_number": condition_number,
    }
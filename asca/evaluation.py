import logging
import pathlib as pl
import time
from dataclasses import dataclass

import h5py
import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import LinearOperator, cg, eigsh, factorized
from scipy.linalg import eigvalsh, eigvals

from . import utils
from .schur_complement import schur_complement
from .asca import (
    HDF5_ADJACENCY_MATRIX,
    HDF5_APPROXIMATION_MATRIX,
    HDF5_COARSE_COUNT,
    HDF5_SUBGRAPH_SIZE,
)

logger = logging.getLogger(__name__)

EVALUATION_OUTPUT_DIRECTORY = "evaluation"
SCHUR_CACHE_DIRECTORY = "schur_cache"
ITERATION_PREFIX = "iteration"
EVALUATION_OUTPUT_SUFFIX = "_evaluation.hdf5"
OUTPUT_VERTEX_COUNT = "vertex_count"
OUTPUT_COARSE_COUNT = "coarse_count"
OUTPUT_MEAN_SUBGRAPH = "mean_subgraph"
OUTPUT_ITERATION_COUNT = "iteration_count"
OUTPUT_INFO = "info"
OUTPUT_ERROR_HISTORY = "error_history"
OUTPUT_RESIDUAL_HISTORY = "residual_history"
OUTPUT_EIGENVALUES = "eigenvalues"
OUTPUT_CONDITION_NUMBER = "condition_number"
NORMALIZATION = 1e-5


@dataclass
class Iteration:
    approximation_matrix: csr_matrix | None = None
    schur_complement_matrix: csr_matrix | None = None
    adjacency_matrix: csr_matrix | None = None
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
            else pl.Path(EVALUATION_OUTPUT_DIRECTORY)
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
                logger.info("Iteration %s: starting CG evaluation", iteration)
                output.update(cg_evaluation(iteration_data))
                logger.info("Iteration %s: finished CG evaluation", iteration)
            if config.evaluate_eigv:
                logger.info("Iteration %s: starting eigenvalue evaluation", iteration)
                output.update(eigsh_evaluation(iteration_data))
                logger.info("Iteration %s: finished eigenvalue evaluation", iteration)

            output[OUTPUT_VERTEX_COUNT] = iteration_data.vertex_count
            output[OUTPUT_COARSE_COUNT] = iteration_data.coarse_count
            output[OUTPUT_MEAN_SUBGRAPH] = iteration_data.subgraph_size_mean

            logger.info("Iteration %s: writing evaluation output", iteration)
            utils.write_data(output_file, f"{ITERATION_PREFIX}{iteration}", output)

    logger.info("Evaluation finished")


def _get_iterations(input_file: pl.Path) -> list[int]:
    with h5py.File(input_file, mode="r") as file:
        iterations = []
        for key in file.keys():
            if key.startswith(ITERATION_PREFIX):
                iterations.append(int(key.replace(ITERATION_PREFIX, "")))

    return list(sorted(iterations))


def _get_output_file(config: EvaluatorConfig, input_file: pl.Path) -> pl.Path:
    output_name = f"{input_file.stem}{EVALUATION_OUTPUT_SUFFIX}"
    if config.output_directory is not None:
        return (
            config.output_directory / output_name
            if config.output_directory is not None
            else output_name
        )

    return input_file.with_name(output_name)


def _normalize_matrix(matrix: csr_matrix) -> csr_matrix:
    return matrix + eye(matrix.shape[0], format="csr") * NORMALIZATION


def _get_matrices(
    input_file: pl.Path,
    iteration: int,
) -> Iteration:
    iteration_data = Iteration(iteration=iteration)

    logger.info("Iteration %s: reading matrices", iteration)

    with h5py.File(input_file, mode="r") as file:
        iteration_group = file[f"{ITERATION_PREFIX}{iteration}"]

        iteration_data.adjacency_matrix = utils.read_csr_matrix(
            iteration_group[HDF5_ADJACENCY_MATRIX]
        )
        iteration_data.vertex_count = iteration_data.adjacency_matrix.shape[0]

        iteration_data.approximation_matrix = utils.read_csr_matrix(
            iteration_group[HDF5_APPROXIMATION_MATRIX]
        )
        iteration_data.coarse_count = utils.read_int(iteration_group, HDF5_COARSE_COUNT)
        iteration_data.subgraph_size_mean = utils.read_int(
            iteration_group, HDF5_SUBGRAPH_SIZE
        )

    iteration_data.approximation_matrix = _normalize_matrix(
        iteration_data.approximation_matrix
    )

    cache_key = _schur_cache_key(
        iteration_data.adjacency_matrix, iteration_data.coarse_count
    )

    schur = None

    for file in pl.Path(SCHUR_CACHE_DIRECTORY).glob("*.hdf5"):
        if cache_key in file.stem:
            with h5py.File(file) as schur_cache:
                schur = utils.read_csr_matrix(schur_cache)
                continue

    if schur is None:
        logger.info("Iteration %s: computing exact Schur complement", iteration)
        schur = schur_complement(
            iteration_data.adjacency_matrix, iteration_data.coarse_count
        )
        logger.info("Iteration %s: exact Schur complement computed", iteration)
        utils.create_folder(SCHUR_CACHE_DIRECTORY)
        with h5py.File(f"{SCHUR_CACHE_DIRECTORY}/{cache_key}.hdf5", mode="w") as file:
            utils.write_csr_matrix(file, schur)
    else:
        logger.info("Iteration %s: loaded exact Schur complement from cache", iteration)

    iteration_data.schur_complement_matrix = _normalize_matrix(schur)

    return iteration_data


def _schur_cache_key(
    adjacency_matrix: csr_matrix,
    coarse_count: int,
) -> tuple:
    return str(hash((adjacency_matrix.shape, adjacency_matrix.nnz, coarse_count)))


def cg_evaluation(iteration_data: Iteration):
    logger.info(
        "Iteration %s: preparing CG evaluation inputs", iteration_data.iteration
    )
    ones = np.ones(iteration_data.approximation_matrix.shape[0])
    ones = ones / np.linalg.norm(ones)

    np.random.seed(42)

    x_exact = np.random.rand(iteration_data.approximation_matrix.shape[0])
    x_exact = x_exact - np.dot(ones, x_exact) * ones

    rhs = iteration_data.schur_complement_matrix @ x_exact
    rhs = rhs - np.dot(ones, rhs) * ones

    solve_m = factorized(iteration_data.approximation_matrix.tocsc())

    linear_op = LinearOperator(
        shape=iteration_data.approximation_matrix.shape,
        matvec=solve_m,
        dtype=np.float64,
    )

    iteration_count = 0
    error_history = []
    residual_history = []

    def cg_callback(x_current):
        nonlocal iteration_count, error_history
        iteration_count += 1
        x_current = x_current - np.dot(ones, x_current) * ones
        error_history.append(np.linalg.norm(x_exact - x_current))
        residual_history.append(
            np.linalg.norm(rhs - iteration_data.schur_complement_matrix @ x_current)
        )

    logger.info("Iteration %s: starting CG solve", iteration_data.iteration)
    start_time = time.perf_counter()
    _, info = cg(
        A=iteration_data.schur_complement_matrix,
        M=linear_op,
        b=rhs,
        rtol=1e-10,
        callback=cg_callback,
    )
    logger.info(
        "Iteration %s: CG solve finished in %fs with info=%s after %s iterations",
        iteration_data.iteration,
        time.perf_counter() - start_time,
        info,
        iteration_count,
    )

    return {
        OUTPUT_ITERATION_COUNT: iteration_count,
        OUTPUT_INFO: info,
        OUTPUT_ERROR_HISTORY: error_history,
        OUTPUT_RESIDUAL_HISTORY: residual_history,
    }


def eigsh_evaluation(iteration_data: Iteration):
    logger.info(
        "Iteration %s: computing largest generalized eigenvalue",
        iteration_data.iteration,
    )
    start_time = time.perf_counter()
    largest_eigenvalue = eigsh(
        A=iteration_data.schur_complement_matrix,
        M=iteration_data.approximation_matrix,
        k=1,
        tol=1e-2,
        which="LA",
        return_eigenvectors=False,
        maxiter=1000,
    )
    logger.info(
        "Iteration %s: largest eigenvalue solve finished in %fs",
        iteration_data.iteration,
        time.perf_counter() - start_time,
    )

    logger.info(
        "Iteration %s: computing smallest generalized eigenvalue",
        iteration_data.iteration,
    )
    start_time = time.perf_counter()
    smallest_eigenvalue = eigsh(
        A=iteration_data.schur_complement_matrix,
        M=iteration_data.approximation_matrix,
        k=1,
        tol=1e-2,
        which="SA",
        return_eigenvectors=False,
        maxiter=1000,
    )
    logger.info(
        "Iteration %s: smallest eigenvalue solve finished in %fs",
        iteration_data.iteration,
        time.perf_counter() - start_time,
    )

    condition_number = max(largest_eigenvalue) / min(smallest_eigenvalue)

    eigenvalues = 0

    start_time = time.perf_counter()

    if iteration_data.schur_complement_matrix.shape[0] < 4000:
        logger.info(
            "Iteration %s: computing all eigenvalues.",
            iteration_data.iteration,
        )
        eigenvalues = eigvalsh(
            a=iteration_data.schur_complement_matrix.todense(),
            b=iteration_data.approximation_matrix.todense()
        )
    else:
        k = min(6, iteration_data.schur_complement_matrix.shape[0] - 1)

        logger.info(
            "Iteration %s: computing eigenvalue spectrum sample with k=%s",
            iteration_data.iteration,
            k,
        )
        eigenvalues = eigsh(
            A=iteration_data.schur_complement_matrix,
            M=iteration_data.approximation_matrix,
            k=k,
            return_eigenvectors=False,
        )

    logger.info(
        "Iteration %s: eigenvalue spectrum solve finished in %fs",
        iteration_data.iteration,
        time.perf_counter() - start_time,
    )

    return {
        OUTPUT_EIGENVALUES: eigenvalues,
        OUTPUT_CONDITION_NUMBER: condition_number,
    }

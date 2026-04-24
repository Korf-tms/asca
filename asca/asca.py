import time
import logging
from dataclasses import dataclass, field
import pathlib as pl
from typing import Callable

from scipy.sparse import csr_matrix
from joblib import Parallel, delayed
import numpy as np
import h5py

from .graph import OriginalGraph
from . import select_coarse
from . import create_subgraph
from . import schur_complement
from . import graph_io
from . import utils

DATA_FOLDER = "data"
DEFAULT_OUTPUT_FILE = "evaluation.hdf5"
ITERATION_PREFIX = "iteration"
HDF5_ADJACENCY_MATRIX = "adj_matrix"
HDF5_APPROXIMATION_MATRIX = "approximation"
HDF5_SUBGRAPH_SIZE = "subgraph_size"
HDF5_COARSE_COUNT = "coarse_count"
DEFAULT_SIZE_ARGUMENT = {"size": 1}
DEFAULT_COARSE_SELECTION_METHOD = "mis"
DEFAULT_SUBGRAPH_CREATION_METHOD = "depth"
COARSE_METHOD_MIS = "mis"
COARSE_METHOD_MIS_DEGREE_ASC = "mis_degree_asc"
COARSE_METHOD_MIS_DEGREE_DESC = "mis_degree_desc"
COARSE_METHOD_MIS_STRENGTH_ASC = "mis_strength_asc"
COARSE_METHOD_MIS_STRENGTH_DESC = "mis_strength_desc"
COARSE_METHOD_MOORE = "moore"
SUBGRAPH_METHOD_DEPTH = "depth"
SUBGRAPH_METHOD_MOORE_ALL = "moore_all"
SUBGRAPH_METHOD_MOORE_COARSE = "moore_coarse"
SUBGRAPH_METHOD_MACROSTRUCTURE = "macrostructure"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    coarse_selection_method_name: str
    coarse_selection_method: Callable
    coarse_selection_method_arguments: dict
    subgraph_creation_method_name: str
    subgraph_creation_method: Callable
    subgraph_creation_method_arguments: dict
    output_file: pl.Path


@dataclass
class AscaConfig:
    filename: pl.Path | str
    coarse_selection_method: str | list[str] = DEFAULT_COARSE_SELECTION_METHOD
    coarse_selection_method_arguments: dict | list[dict] | None = None
    subgraph_creation_method: str | list[str] = DEFAULT_SUBGRAPH_CREATION_METHOD
    subgraph_creation_method_arguments: dict | list[dict] | None = None
    output_file: pl.Path | str | None = None
    iterations: int = 1
    path: pl.Path = field(init=False)
    base_output_file: pl.Path = field(init=False)
    config: list[Config] = field(init=False)

    coarse_selection_methods = {
        COARSE_METHOD_MIS: select_coarse.mis,
        COARSE_METHOD_MIS_DEGREE_ASC: select_coarse.mis_degree_asc,
        COARSE_METHOD_MIS_DEGREE_DESC: select_coarse.mis_degree_desc,
        COARSE_METHOD_MIS_STRENGTH_ASC: select_coarse.mis_strength_asc,
        COARSE_METHOD_MIS_STRENGTH_DESC: select_coarse.mis_strength_desc,
        COARSE_METHOD_MOORE: select_coarse.moore,
    }
    create_subgraphs_methods = {
        SUBGRAPH_METHOD_DEPTH: create_subgraph.depth,
        SUBGRAPH_METHOD_MOORE_ALL: create_subgraph.moore_neighborhood_all,
        SUBGRAPH_METHOD_MOORE_COARSE: create_subgraph.moore_neighborhood_around_coarse,
        SUBGRAPH_METHOD_MACROSTRUCTURE: create_subgraph.macrostructures,
    }

    @staticmethod
    def _ensure_list(value):
        return value if isinstance(value, list) else [value]

    @staticmethod
    def _format_arguments(arguments: dict) -> str:
        return "_".join(f"{key}{value}" for key, value in sorted(arguments.items()))

    def __post_init__(self):
        utils.create_folder(DATA_FOLDER)

        coarse_selection_method_arguments = self.coarse_selection_method_arguments
        if coarse_selection_method_arguments is None:
            coarse_selection_method_arguments = DEFAULT_SIZE_ARGUMENT

        subgraph_creation_method_arguments = self.subgraph_creation_method_arguments
        if subgraph_creation_method_arguments is None:
            subgraph_creation_method_arguments = DEFAULT_SIZE_ARGUMENT

        self.path = pl.Path(self.filename)
        self.base_output_file = (
            pl.Path(DATA_FOLDER) / DEFAULT_OUTPUT_FILE
            if self.output_file is None
            else pl.Path(self.output_file)
        )

        coarse_selection_method = self._ensure_list(self.coarse_selection_method)
        coarse_selection_method_arguments = self._ensure_list(
            coarse_selection_method_arguments
        )
        subgraph_creation_method = self._ensure_list(self.subgraph_creation_method)
        subgraph_creation_method_arguments = self._ensure_list(
            subgraph_creation_method_arguments
        )

        if not (
            len(coarse_selection_method)
            == len(coarse_selection_method_arguments)
            == len(subgraph_creation_method)
            == len(subgraph_creation_method_arguments)
        ):
            raise ValueError(
                "Method lists and argument lists must have the same length."
            )

        self.config = [
            Config(
                coarse_selection_method_name=coarse_method_name,
                coarse_selection_method=self.coarse_selection_methods[
                    coarse_method_name
                ],
                coarse_selection_method_arguments=coarse_method_arguments,
                subgraph_creation_method_name=subgraph_method_name,
                subgraph_creation_method=self.create_subgraphs_methods[
                    subgraph_method_name
                ],
                subgraph_creation_method_arguments=subgraph_method_arguments,
                output_file=self.base_output_file.with_name(
                    f"{self.base_output_file.stem}_{coarse_method_name}_{self._format_arguments(coarse_method_arguments)}_{subgraph_method_name}_{self._format_arguments(subgraph_method_arguments)}{self.base_output_file.suffix}"
                ),
            )
            for (
                coarse_method_name,
                coarse_method_arguments,
                subgraph_method_name,
                subgraph_method_arguments,
            ) in zip(
                coarse_selection_method,
                coarse_selection_method_arguments,
                subgraph_creation_method,
                subgraph_creation_method_arguments,
            )
        ]


def run_approximation(config: AscaConfig):
    logger.info(
        "Starting ASCA for %s config(s), %s iteration(s) each",
        len(config.config),
        config.iterations,
    )
    logger.info("ASCA input: %s", config.path)

    for step in config.config:
        logger.info(
            "Starting config: coarse=%s %s, subgraph=%s %s",
            step.coarse_selection_method_name,
            step.coarse_selection_method_arguments,
            step.subgraph_creation_method_name,
            step.subgraph_creation_method_arguments,
        )
        logger.info("ASCA output: %s", step.output_file)
        config_start_time = time.perf_counter()

        try:
            current_iteration = 0
            approximation_matrix = 0

            logger.info("Reading input graph")
            start_time = time.perf_counter()
            current_graph: OriginalGraph = graph_io.from_file(
                path=config.path, cls=OriginalGraph
            )
            logger.info(
                "Input graph read in %.2fs (%s vertices)",
                time.perf_counter() - start_time,
                len(current_graph.vertex_list),
            )

            for _ in range(config.iterations):
                logger.info("Iteration %s: starting approximation", current_iteration)
                iteration_start_time = time.perf_counter()

                approximation_matrix = calculate_approximation(
                    current_graph,
                    coarse_selection_method=step.coarse_selection_method,
                    coarse_selection_method_arguments=step.coarse_selection_method_arguments,
                    subgraph_creation_method=step.subgraph_creation_method,
                    subgraph_creation_method_arguments=step.subgraph_creation_method_arguments,
                )
                subgraph_mean = round(
                    np.mean(
                        [
                            len(subgraph.vertex_list)
                            for subgraph in current_graph.subgraph_list
                        ]
                    )
                )
                logger.info(
                    "Iteration %s: mean subgraph size is %s",
                    current_iteration,
                    subgraph_mean,
                )
                logger.info("Iteration %s: writing output", current_iteration)
                start_time = time.perf_counter()
                store_iteration(
                    step.output_file,
                    current_iteration,
                    current_graph.to_adj_matrix(sorting=current_graph.vertex_sort),
                    approximation_matrix,
                    subgraph_mean,
                    current_graph.coarse_vertices_count,
                )
                logger.info(
                    "Iteration %s: output written in %.2fs",
                    current_iteration,
                    time.perf_counter() - start_time,
                )

                logger.info("Iteration %s: preparing next graph", current_iteration)
                start_time = time.perf_counter()
                approximation_matrix = laplacian_to_adj_mat(approximation_matrix)
                current_graph = graph_io.from_coo(
                    coo_mat=approximation_matrix.tocoo(), cls=OriginalGraph
                )
                logger.info(
                    "Iteration %s: next graph prepared in %.2fs",
                    current_iteration,
                    time.perf_counter() - start_time,
                )
                logger.info(
                    "Iteration %s: done in %.2fs",
                    current_iteration,
                    time.perf_counter() - iteration_start_time,
                )
                current_iteration += 1

        except Exception as e:
            logger.exception(
                "ASCA failed for coarse=%s %s, subgraph=%s %s",
                step.coarse_selection_method_name,
                step.coarse_selection_method_arguments,
                step.subgraph_creation_method_name,
                step.subgraph_creation_method_arguments,
            )
            break

        logger.info(
            "Config finished in %.2fs",
            time.perf_counter() - config_start_time,
        )

    logger.info("ASCA finished")


def laplacian_to_adj_mat(laplacian: csr_matrix) -> csr_matrix:
    adj_matrix: csr_matrix = laplacian.copy()
    adj_matrix = -adj_matrix
    adj_matrix.setdiag(0)
    adj_matrix.eliminate_zeros()
    return adj_matrix


def store_iteration(
    path: str | pl.Path,
    iteration: int,
    adj_matrix: csr_matrix,
    approximation: csr_matrix,
    mean_subgraph_count: int,
    coarse_count: int,
):
    with h5py.File(path, mode="a") as file:
        iteration_group = file.require_group(f"{ITERATION_PREFIX}{iteration}")
        adj_mat_group = iteration_group.require_group(HDF5_ADJACENCY_MATRIX)
        utils.write_csr_matrix(adj_mat_group, adj_matrix)
        approximation_group = iteration_group.require_group(HDF5_APPROXIMATION_MATRIX)
        utils.write_csr_matrix(approximation_group, approximation)
        iteration_group.create_dataset(HDF5_SUBGRAPH_SIZE, data=mean_subgraph_count)
        iteration_group.create_dataset(HDF5_COARSE_COUNT, data=coarse_count)


def calculate_approximation(
    in_graph: OriginalGraph,
    coarse_selection_method,
    coarse_selection_method_arguments: dict,
    subgraph_creation_method,
    subgraph_creation_method_arguments: dict,
):
    degrees = [(x, len(x.adj)) for x in in_graph.vertex_list]
    logger.info(
        "Starting approximation: vertices=%s, min degree=%s, max degree=%s",
        len(in_graph.vertex_list),
        min(degrees, key=lambda x: x[1])[1],
        max(degrees, key=lambda x: x[1])[1],
    )

    logger.info("Selecting coarse vertices")
    start_time = time.perf_counter()
    coarse_selection_method(in_graph, **coarse_selection_method_arguments)
    logger.info(
        "Coarse selection took %.2fs (%s coarse vertices)",
        time.perf_counter() - start_time,
        in_graph.coarse_vertices_count,
    )

    logger.info("Creating subgraphs")
    start_time = time.perf_counter()
    subgraph_creation_method(in_graph, **subgraph_creation_method_arguments)
    logger.info(
        "Subgraph creation took %.2fs (%s subgraphs)",
        time.perf_counter() - start_time,
        len(in_graph.subgraph_list),
    )

    logger.info("Updating edge multiplicities")
    start_time = time.perf_counter()
    in_graph.update_edge_multiplicities()
    logger.info(
        "Edge multiplicity calculation took %.2fs",
        time.perf_counter() - start_time,
    )

    logger.info("Computing approximation matrix")
    start_time = time.perf_counter()
    approximation_matrix = csr_matrix(
        (in_graph.coarse_vertices_count, in_graph.coarse_vertices_count),
        dtype=np.float64,
    )

    generator = Parallel(n_jobs=-1, prefer="threads", return_as="generator_unordered")(
        delayed(schur_complement.get_contribution)(subgraph)
        for subgraph in in_graph.subgraph_list
    )

    for contribution in generator:
        approximation_matrix += contribution
    logger.info(
        "Approximation matrix took %.2fs",
        time.perf_counter() - start_time,
    )
    return approximation_matrix

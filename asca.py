import time
import logging
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from scipy.sparse import csr_matrix
from joblib import Parallel, delayed
import numpy as np
import h5py

from graph import OriginalGraph
import select_coarse
import create_subgraph
import schur_complement
import graph_io
import utils

DATA_FOLDER = "data"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    coarse_selection_method_name: str
    coarse_selection_method: Callable
    coarse_selection_method_arguments: dict
    subgraph_creation_method_name: str
    subgraph_creation_method: Callable
    subgraph_creation_method_arguments: dict
    output_file: Path


@dataclass
class AscaConfig:
    filename: Path | str
    coarse_selection_method: str | list[str] = "mis"
    coarse_selection_method_arguments: dict | list[dict] | None = None
    subgraph_creation_method: str | list[str] = "depth"
    subgraph_creation_method_arguments: dict | list[dict] | None = None
    output_file: Path | str | None = None
    iterations: int = 1
    path: Path = field(init=False)
    base_output_file: Path = field(init=False)
    config: list[Config] = field(init=False)

    coarse_selection_methods = {
        "mis": select_coarse.mis,
        "mis_degree_asc": select_coarse.mis_degree_asc,
        "mis_degree_desc": select_coarse.mis_degree_desc,
        "mis_strength_asc": select_coarse.mis_strength_asc,
        "mis_strength_desc": select_coarse.mis_strength_desc,
        "moore": select_coarse.moore,
    }
    create_subgraphs_methods = {
        "depth": create_subgraph.create_subgraphs_depth,
        "moore_all": create_subgraph.moore_neighborhood_all,
        "moore_coarse": create_subgraph.moore_neighborhood_around_coarse,
        "macrostructure": create_subgraph.create_subgraphs_macrostructures,
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
            coarse_selection_method_arguments = {"size": 1}

        subgraph_creation_method_arguments = self.subgraph_creation_method_arguments
        if subgraph_creation_method_arguments is None:
            subgraph_creation_method_arguments = {"size": 1}

        self.path = Path(self.filename)
        self.base_output_file = (
            Path(f"{DATA_FOLDER}/evaluation.hdf5")
            if self.output_file is None
            else Path(self.output_file)
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

    for step in config.config:
        try:
            current_iteration = 0
            approximation_matrix = 0

            current_graph: OriginalGraph = graph_io.from_file(
                path=config.path, cls=OriginalGraph
            )

            for _ in range(config.iterations):
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
                store_iteration(
                    step.output_file,
                    current_iteration,
                    current_graph.to_adj_matrix(sorting=current_graph.vertex_sort),
                    approximation_matrix,
                    subgraph_mean,
                    current_graph.coarse_vertices_count,
                )

                approximation_matrix = laplacian_to_adj_mat(approximation_matrix)
                current_graph = graph_io.from_coo(
                    coo_mat=approximation_matrix.tocoo(), cls=OriginalGraph
                )
                current_iteration += 1

        except Exception as e:
            print(
                f"Error at {step.coarse_selection_method_name}: {step.coarse_selection_method_arguments}, {step.subgraph_creation_method_name}: {step.subgraph_creation_method_arguments}"
            )
            print(e)


def laplacian_to_adj_mat(laplacian: csr_matrix) -> csr_matrix:
    adj_matrix: csr_matrix = laplacian.copy()
    adj_matrix = -adj_matrix
    adj_matrix.setdiag(0)
    adj_matrix.eliminate_zeros()
    return adj_matrix


def store_iteration(
    path: str | Path,
    iteration: int,
    adj_matrix: csr_matrix,
    approximation: csr_matrix,
    mean_subgraph_count: int,
    coarse_count: int,
):
    with h5py.File(path, mode="a") as file:
        iteration_group = file.require_group(f"iteration{iteration}")
        adj_mat_group = iteration_group.require_group("adj_matrix")
        utils.store_csr_matrix(adj_mat_group, adj_matrix)
        approximation_group = iteration_group.require_group("approximation")
        utils.store_csr_matrix(approximation_group, approximation)
        iteration_group.create_dataset("subgraph_size", data=mean_subgraph_count)
        iteration_group.create_dataset("coarse_count", data=coarse_count)


def calculate_approximation(
    in_graph: OriginalGraph,
    coarse_selection_method,
    coarse_selection_method_arguments: dict,
    subgraph_creation_method,
    subgraph_creation_method_arguments: dict,
):
    degrees = [(x, len(x.adj)) for x in in_graph.vertex_list]
    logger.info(
        f"--Starting approximation, size {len(in_graph.vertex_list)}, min degree {min(degrees, key=lambda x: x[1])}, max degree {max(degrees, key=lambda x: x[1])}"
    )

    start_time = time.time()
    coarse_selection_method(in_graph, **coarse_selection_method_arguments)
    logger.info(f"Coarse selection took {time.time() - start_time}s")

    start_time = time.time()
    subgraph_creation_method(in_graph, **subgraph_creation_method_arguments)
    logger.info(f"Graph creation took {time.time() - start_time}s")

    start_time = time.time()
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
    logger.info(f"Approximation calculation took {time.time() - start_time}s")
    return approximation_matrix

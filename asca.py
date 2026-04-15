import time
import logging

from scipy.sparse import csr_matrix
from joblib import Parallel, delayed
import pathlib as pl
import numpy as np
import h5py

from graph import OriginalGraph
import select_coarse
import create_subgraph
import schur_complement
import graph_io
import utils

LOG_FOLDER = "logs"
DATA_FOLDER = "data"

logger = logging.getLogger(__name__)

class Asca:
    def __init__(
        self,
        filename,
        coarse_selection_method="mis",
        coarse_selection_method_arguments={"size": 1},
        create_subgraphs_method="depth",
        create_subgraphs_method_arguments={"max_depth": 2},
        output_file=None,
        store_contributions=False,
        iterations=1,
    ):

        utils.create_folder(DATA_FOLDER)

        self.path = filename
        self.filename = pl.Path(filename).stem
        self.output_file = utils.get_unique_path(
            self.filename, output_file=output_file, data_folder=DATA_FOLDER, name="data"
        )
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
        self.coarse_selection_method = coarse_selection_methods[coarse_selection_method]
        self.coarse_selection_method_arguments = coarse_selection_method_arguments
        self.create_subgraphs_method = create_subgraphs_methods[create_subgraphs_method]
        self.create_subgraphs_method_arguments = create_subgraphs_method_arguments
        self.iterations = iterations

    def run_approximation(self):
        current_graph: OriginalGraph = graph_io.from_file(
            path=self.path, cls=OriginalGraph
        )

        current_iteration = 0
        Q = 0

        for _ in range(self.iterations):

            # calculate ASCA
            Q: csr_matrix = self.calculate_approximation(current_graph)

            with h5py.File(self.output_file, mode="a") as file:
                iteration_group = file.require_group(f"iteration{current_iteration}")
                adj_mat_group = iteration_group.require_group("adj_matrix")
                utils.store_csr_matrix(
                    adj_mat_group,
                    current_graph.to_adj_matrix(sorting=current_graph.vertex_sort),
                )
                iteration_group.create_dataset(
                    "coarse_count", data=current_graph.coarse_vertices_count
                )

            # getting the adj matrix out of the Laplacian
            Q = -Q
            Q.setdiag(0)
            Q.eliminate_zeros()
            current_graph = graph_io.from_coo(coo_mat=Q.tocoo(), cls=OriginalGraph)

            current_iteration += 1

        # store last iteration, which has no coarse vertices
        with h5py.File(self.output_file, mode="a") as file:
            iteration_group = file.require_group(f"iteration{current_iteration}")
            adj_mat_group = iteration_group.require_group("adj_matrix")
            utils.store_csr_matrix(adj_mat_group, Q)

    def calculate_approximation(self, in_graph: OriginalGraph):
        degrees = [(x, len(x.adj)) for x in in_graph.vertex_list]
        logger.info(f"--Starting approximation, size {len(in_graph.vertex_list)}, min degree {min(degrees, key=lambda x: x[1])}, max degree {max(degrees, key=lambda x: x[1])}")

        # select coarse vertices
        start_time = time.time()
        self.coarse_selection_method(in_graph, **self.coarse_selection_method_arguments)
        logger.info(f"Coarse selection took {time.time() - start_time}s")

        # create subgraphs
        start_time = time.time()
        self.create_subgraphs_method(in_graph, **self.create_subgraphs_method_arguments)
        logger.info(f"Graph creation took {time.time() - start_time}s")

        start_time = time.time()
        Q = csr_matrix(
            (in_graph.coarse_vertices_count, in_graph.coarse_vertices_count),
            dtype=np.float64,
        )

        # add together subgraph contributions
        generator = Parallel(
            n_jobs=-1, prefer="threads", return_as="generator_unordered"
        )(
            delayed(schur_complement.get_contribution)(subgraph)
            for subgraph in in_graph.subgraph_list
        )

        for contribution in generator:
            Q += contribution
        logger.info(f"Approximation calculation took {time.time() - start_time}s")
        return Q

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cgs, cg
from joblib import Parallel, delayed
from datetime import datetime


import numpy as np
import h5py

import sys
import graph
import utils
import time
import logging

logger = logging.getLogger(__name__)

LOG_FOLDER = "logs"
DATA_FOLDER = "data"

class Asca:
    def __init__(self, 
                 filename, 
                 coarse_selection_method="mis", 
                 coarse_selection_method_arguments={"size":1}, 
                 create_subgraphs_method="depth", 
                 create_subgraphs_method_arguments={"max_depth":2},
                 store_contributions=False,
                 iterations=1):
        
        utils.clear_folder_or_create(LOG_FOLDER)
        utils.clear_folder_or_create(DATA_FOLDER)

        now = datetime.now().strftime("%S-%M-%H_%d_%m_%y_log")
        logging.basicConfig(filename=f"{LOG_FOLDER}/{now}.log", filemode='w', level=logging.INFO)

        self.filename = filename
        self.coarse_selection_method = coarse_selection_method
        self.coarse_selection_method_arguments = coarse_selection_method_arguments
        self.create_subgraphs_method = create_subgraphs_method
        self.create_subgraphs_method_arguments = create_subgraphs_method_arguments
        self.iterations = iterations
        self.current_iteration = 0
    
    def calculate_subgraph_contribution(self, sub_graph):
        mapping = sub_graph.local_to_global_mapping()
        schur_complement = sub_graph.local_schur_complement()
        temp = mapping @ schur_complement @ mapping.T
        return temp
    
    def run_approximation(self):
        current_graph = graph.UniversalGraph.from_file(self.filename)

        for i in range(self.iterations):

            coarse_selection_methods = {
                "mis":current_graph.select_coarse_mis,
                "moore":current_graph.select_coarse_moore_neighborhood
            }
            create_subgraphs_methods = {
                "depth":current_graph.create_subgraphs_depth,
                "moore_all":current_graph.create_subgraphs_moore_neighborhood_all,
                "moore_coarse":current_graph.create_subgraphs_moore_neighborhood_around_coarse
            }

            logging.info(f"ASCA Iteration {self.current_iteration} current size: {len(current_graph.vertex_list)}")
            with h5py.File(f"{DATA_FOLDER}/data.hdf5", mode="a") as file:
                group = file.require_group(f"iteration{self.current_iteration}")
                group.create_dataset(f"adj_mat", data=current_graph.vertex_list_to_adj_matrix(current_graph.vertex_list))

            Q = self.calculate_approximation(
                current_graph, 
                coarse_selection_methods[self.coarse_selection_method],
                create_subgraphs_methods[self.create_subgraphs_method])

            with h5py.File(f"{DATA_FOLDER}/data.hdf5", mode="a") as file:
                group = file.require_group(f"iteration{self.current_iteration}")
                group.create_dataset(f"approximation", data=Q.todense())#need a way to make this work with sparse matrix

            self.current_iteration += 1

            #getting the adj matrix out of the Laplacian
            Q = abs(Q)
            indexes = (range(Q.shape[0]), range(Q.shape[1]))
            Q[indexes] = 0
            current_graph = graph.UniversalGraph.from_csr(Q)

    def calculate_approximation(self, graph, coarse_selection_method, create_subgraphs_method):
        coarse_selection_method(**self.coarse_selection_method_arguments)
        start_time = time.time()
        logging.info(f"Coarse vertex selection took {time.time() - start_time} seconds.")

        start_time = time.time()
        create_subgraphs_method(**self.create_subgraphs_method_arguments)
        logging.info(f"Subgraph creation took {time.time() - start_time} seconds.")

        start_time = time.time()
        l = graph.coarse_vertices_count
        Q = csr_matrix((l, l), dtype=np.float64)
        
        generator = Parallel(
            n_jobs=-1, 
            prefer="threads",
            return_as="generator"
        )(
            delayed(self.calculate_subgraph_contribution)
            (subgraph) for subgraph in graph.get_subgraphs()
        )
        for contribution in generator:
            Q += contribution
        logging.info(f"Calculation took {time.time() - start_time} seconds.")
        return Q
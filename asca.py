from scipy.sparse import csr_matrix
from joblib import Parallel, delayed
from datetime import datetime

import pathlib as pl
import numpy as np
import h5py

import graph
import utils
import time
import logging

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

        self.path = filename
        self.filename = pl.Path(filename).stem
        self.coarse_selection_method = coarse_selection_method
        self.coarse_selection_method_arguments = coarse_selection_method_arguments
        self.create_subgraphs_method = create_subgraphs_method
        self.create_subgraphs_method_arguments = create_subgraphs_method_arguments
        self.iterations = iterations
    
    def store_csr_matrix(self, gorup : h5py.Group, matrix : csr_matrix):
        gorup.create_dataset("data", data=matrix.data)
        gorup.create_dataset("indices", data=matrix.indices)
        gorup.create_dataset("indptr", data=matrix.indptr)
        gorup.create_dataset("shape", data=matrix.shape)

    def run_approximation(self):
        current_graph = graph.Graph(path=self.path)

        for i in range(1, self.iterations + 1):

            coarse_selection_methods = {
                "mis":current_graph.select_coarse_mis,
                "moore":current_graph.select_coarse_moore_neighborhood
            }
            create_subgraphs_methods = {
                "depth":current_graph.create_subgraphs_depth,
                "moore_all":current_graph.create_subgraphs_moore_neighborhood_all,
                "moore_coarse":current_graph.create_subgraphs_moore_neighborhood_around_coarse,
                "macrostructure":current_graph.create_subgraphs_macrostructures
            }

            logging.info(f"ASCA Iteration {i} current size: {len(current_graph.vertex_list)}")

            Q : csr_matrix = self.calculate_approximation(
                current_graph, 
                coarse_selection_methods[self.coarse_selection_method],
                create_subgraphs_methods[self.create_subgraphs_method])

            with h5py.File(f"{DATA_FOLDER}/{self.filename}_data.hdf5", mode="a") as file:
                if i == 1:
                    adj_mat = current_graph.vertex_list_to_adj_matrix()
                    adj_matrix_group = file.require_group(f"adj_matrix")
                    self.store_csr_matrix(adj_matrix_group, adj_mat)
                    adj_matrix_group.create_dataset("coarse_count", data=current_graph.coarse_vertices_count)
                iteration_group = file.require_group(f"iteration{i}")
                self.store_csr_matrix(iteration_group, Q)

            #getting the adj matrix out of the Laplacian
            Q = abs(Q)
            Q.setdiag(0)
            current_graph = graph.Graph(csr_matrix=Q)

    def calculate_approximation(self, in_graph : graph.Graph, coarse_selection_method, create_subgraphs_method):
        
        #select coarse vertices
        start_time = time.time()
        coarse_selection_method(**self.coarse_selection_method_arguments)
        logging.info(f"Coarse vertex selection took {time.time() - start_time} seconds.")

        #create subgraphs
        start_time = time.time()
        create_subgraphs_method(**self.create_subgraphs_method_arguments)
        logging.info(f"Subgraph creation took {time.time() - start_time} seconds.")

        start_time = time.time()
        Q = csr_matrix((in_graph.coarse_vertices_count, in_graph.coarse_vertices_count), dtype=np.float64)
        
        #add together subgraph contributions
        generator = Parallel(
            n_jobs=-1, 
            prefer="threads",
            return_as="generator_unordered"
        )(
            delayed(subgraph.get_contribution)()
            for subgraph in in_graph.subgraph_list
        )

        for contribution in generator:
            Q += contribution
        logging.info(f"Calculation took {time.time() - start_time} seconds.")
        return Q
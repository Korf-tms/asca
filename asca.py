from scipy.sparse import csr_matrix
from joblib import Parallel, delayed
from datetime import datetime

import numpy as np
import h5py

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
    
    def run_approximation(self):
        current_graph = graph.Graph(path=self.filename)

        for i in range(self.iterations):

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

            logging.info(f"ASCA Iteration {self.current_iteration} current size: {len(current_graph.vertex_list)}")

            Q : csr_matrix = self.calculate_approximation(
                current_graph, 
                coarse_selection_methods[self.coarse_selection_method],
                create_subgraphs_methods[self.create_subgraphs_method])

            with h5py.File(f"{DATA_FOLDER}/data.hdf5", mode="a") as file:
                iteration_group = file.require_group(f"iteration{i}")
                approximation_group =  iteration_group.require_group(f"approximation")
                approximation_group.create_dataset(f"data", data=Q.data)
                approximation_group.create_dataset(f"indices", data=Q.indices)
                approximation_group.create_dataset(f"indptr", data=Q.indptr)

            #getting the adj matrix out of the Laplacian
            Q = abs(Q)
            Q.setdiag(0)
            current_graph = graph.Graph(csr_matrix=Q)

    def calculate_approximation(self, in_graph, coarse_selection_method, create_subgraphs_method):
        
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
            delayed(subgraph.get_contribution)
            (subgraph) for subgraph in in_graph.get_subgraphs()
        )

        for contribution in generator:
            Q += contribution
        logging.info(f"Calculation took {time.time() - start_time} seconds.")

        return Q
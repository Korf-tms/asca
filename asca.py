from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cgs, cg
from joblib import Parallel, delayed

import numpy as np
import pandas as pd

import sys
import graph
import utils
import time

class Asca:
    def __init__(self, filename, iterations=1):
        self.current_graph = graph.UniversalGraph.from_file(filename)
        self.iterations = iterations
        self.current_iteration = 0
        self.cgs_iterations = 0
    
    def calculate_subgraph_contribution(self, sub_graph):
        mapping = sub_graph.local_to_global_mapping()
        schur_complement = sub_graph.local_schur_complement()
        temp = mapping @ schur_complement @ mapping.T
        return temp

    def cgs_callback(self, solution_vector):
        self.cgs_iterations += 1

    def solve_asca(self):
        graph_size = len(self.current_graph.vertex_list)
        print(f"ASCA Iteration {self.current_iteration}\ncurrent size: {graph_size}")

        # selecting coarse vertices
        start_time = time.time()
        self.current_graph.select_coarse_mis(1)
        print(f"Coarse vertex selection took {time.time() - start_time} seconds.")

        # creating subgraphs
        start_time = time.time()
        self.current_graph.create_subgraphs_depth(2)
        print(f"Subgraph creation took {time.time() - start_time} seconds.")

        if graph_size < 200:
            utils.visualize_graph(self.current_graph, name=f"Graph{self.current_iteration}")

        # calculating approximation
        start_time = time.time()
        l = self.current_graph.coarse_vertices_count
        Q = csr_matrix((l, l), dtype=np.float64)
        
        generator = Parallel(
            n_jobs=-1, 
            prefer="threads",
            return_as="generator"
        )(
            delayed(self.calculate_subgraph_contribution)
            (subgraph) for subgraph in self.current_graph.get_subgraphs()
        )
        for contribution in generator:
            Q += contribution
        print(f"Calculation took {time.time() - start_time} seconds.")
        
        schur = self.current_graph.local_schur_complement()

        # cg needs semi positive definite matrix, even small negatives make issues
        approximation = Q + np.eye(Q.shape[0]) * 1e-5
        schur += np.eye(schur.shape[0]) * 1e-5
        
        tolerance = 1e-5
        print(f"Matrix symetry check of approximation with tolerance {tolerance}: {np.allclose(approximation, approximation.T, rtol=tolerance, atol=tolerance)}")
        print(f"Matrix symetry check of schur complement with tolerance {tolerance}: {np.allclose(schur, schur.T, rtol=tolerance, atol=tolerance)}")
        sdp_approximation = np.all(np.linalg.eigvalsh(approximation) >= 0)
        print(f"Positive semi-definite check approximation: {sdp_approximation}")
        spd_schur = np.all(np.linalg.eigvalsh(schur) >= 0)
        print(f"Positive semi-definite check schur complement: {spd_schur}")

        if not sdp_approximation or not spd_schur:
            print("Warning: Matrices are not positive semi-definite. CGS may not converge.")

        self.cgs_iterations = 0

        b = np.random.rand(Q.shape[0], 1)
        print(f"Return value of cgs: {cgs(
            A=schur, 
            M=approximation, 
            b=b,
            callback=self.cgs_callback
            )[1]}")

        print(f"Number of iterations of cgs: {self.cgs_iterations}")

        with pd.HDFStore("data/analysis.hdf5", mode="a") as store:
            store.put(f"analysis/iteration{self.current_iteration}/asca", pd.DataFrame(approximation))
            store.put(f"analysis/iteration{self.current_iteration}/schur_complement", pd.DataFrame(schur))
            store.put(f"analysis/iteration{self.current_iteration}/difference", pd.DataFrame(schur - approximation))
        
        Q = abs(Q)
        indexes = (range(Q.shape[0]), range(Q.shape[1]))
        Q[indexes] = 0
        self.current_graph = graph.GridGraph.from_csr(Q)
        self.current_iteration += 1

if len(sys.argv) != 2:
    print("Usage: python asca.py <path_to_file>")
    sys.exit(1)

utils.clear_folder_or_create("data")
utils.clear_folder_or_create("images")

asca = Asca(sys.argv[1])

for _ in range(1):
    asca.solve_asca()
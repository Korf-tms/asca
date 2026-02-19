from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cgs, cg
from joblib import Parallel, delayed

import numpy as np
import h5py

import sys
import graph
import utils
import time

class Asca:
    def __init__(self, filename, iterations=1):
        self.current_graph = graph.UniversalGraph.from_file(filename)
        self.current_approximation = None
        self.iterations = iterations
        self.current_iteration = 0
        self.cgs_iterations = 0
    
    def calculate_subgraph_contribution(self, sub_graph):
        mapping = sub_graph.local_to_global_mapping()
        schur_complement = sub_graph.local_schur_complement()
        temp = mapping @ schur_complement @ mapping.T
        return temp

    def cgs_callback(self, solution_vector):
        group = self.file.require_group(f"iteration{self.current_iteration}")
        group = group.require_group(f"cgs_iterations")
        group.create_dataset(f"cgs_solution_{self.cgs_iterations}", data=solution_vector)
        self.cgs_iterations += 1
        

    def solve_asca(self):

        if self.current_iteration != 0:
            self.current_approximation = abs(self.current_approximation)
            indexes = (range(self.current_approximation.shape[0]), range(self.current_approximation.shape[1]))
            self.current_approximation[indexes] = 0
            self.current_graph = graph.UniversalGraph.from_csr(self.current_approximation)

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

        self.current_approximation = Q
        self.current_iteration += 1
    
    def evaluate_approximation(self):
        # cg needs semi positive definite matrix, even small negatives make issues
        approximation = self.current_approximation + np.eye(self.current_approximation.shape[0]) * 1e-5
        schur = self.current_graph.local_schur_complement() + np.eye(self.current_approximation.shape[0]) * 1e-5
        
        self.file = h5py.File("data/analysis.hdf5", mode="w")
        group = self.file.require_group(f"iteration{self.current_iteration}")
        group.create_dataset(f"asca", data=approximation)
        group.create_dataset(f"schur_complement", data=schur)
        group.create_dataset(f"difference", data=schur - approximation)

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

        b = np.random.rand(approximation.shape[0], 1)
        print(f"Return value of cgs: {cgs(
            A=schur, 
            M=approximation, 
            b=b,
            callback=self.cgs_callback
            )[1]}")
        self.file.close()
        print(f"Number of iterations of cgs: {self.cgs_iterations}")

if len(sys.argv) != 2:
    print("Usage: python asca.py <path_to_file>")
    sys.exit(1)

utils.clear_folder_or_create("data")
utils.clear_folder_or_create("images")

asca = Asca(sys.argv[1])

for _ in range(2):
    asca.solve_asca()

asca.evaluate_approximation()
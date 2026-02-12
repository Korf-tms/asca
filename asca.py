from scipy.sparse import csr_matrix
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
        self.solutions = list()
        self.iterations = iterations
        self.current_iteration = 0
    
    def solve_asca(self):
        # currently set for grid structures
        print(f"ASCA Iteration {self.current_iteration}\ncurrent size: {len(self.current_graph.vertex_list)}")
        # selecting coarse vertices
        start_time = time.time()
        self.current_graph.select_coarse_mis(1)
        print(f"Coarse vertex selection took {time.time() - start_time} seconds.")
        # creating subgraphs
        start_time = time.time()
        self.current_graph.create_subgraphs_depth(2)
        print(f"Subgraph creation took {time.time() - start_time} seconds.")
        #utils.visualize_graph(self.current_graph, name=f"Graph{self.current_iteration}")
        # initialize approximation matrix
        l = self.current_graph.coarse_vertices_count
        Q = csr_matrix((l, l), dtype=np.float64)
        
        def calculate_subgraph_contribution(sub_graph):
            mapping = sub_graph.local_to_global_mapping()
            schur_complement = sub_graph.local_schur_complement()
            temp = mapping @ schur_complement @ mapping.transpose()
            return temp
        
        start_time = time.time()
        generator = Parallel(
            n_jobs=-1, 
            prefer="threads",
            return_as="generator"
        )(
            delayed(calculate_subgraph_contribution)
            (subgraph) for subgraph in self.current_graph.get_subgraphs()
        )
        for contribution in generator:
            Q += contribution
        print(f"Calculation took {time.time() - start_time} seconds.")
        
        self.solutions.append(Q)
        
        schur = self.current_graph.local_schur_complement()
        main_graph_adj_matrix = self.current_graph.vertex_list_to_adj_matrix(self.current_graph.vertex_list)
        approximation = Q.todense()

        with pd.HDFStore("data/analysis.hdf5", mode="a") as store:
            store.put(f"analysis/iteration{self.current_iteration}/adj_matrix", pd.DataFrame(main_graph_adj_matrix))
            store.put(f"analysis/iteration{self.current_iteration}/asca", pd.DataFrame(approximation))
            store.put(f"analysis/iteration{self.current_iteration}/schur_complement", pd.DataFrame(schur))
            store.put(f"analysis/iteration{self.current_iteration}/difference", pd.DataFrame(schur - approximation))
        
        # getting adj matrix out of laplacian matrix
        approximation = abs(approximation)
        indexes = (range(len(approximation)), range(len(approximation)))
        approximation[indexes] = 0
        self.current_graph = graph.GridGraph.from_csr(Q)
        self.current_iteration += 1

if len(sys.argv) != 2:
    print("Usage: python asca.py <path_to_file>")
    sys.exit(1)

utils.clear_folder_or_create("data")
utils.clear_folder_or_create("images")

asca = Asca(sys.argv[1])

# solving 2 iterations, will crash if graph is too small
for _ in range(2):
    asca.solve_asca()

#hdf5 utils generator!
#argparse
#conjugate gradients kolik iteraci
#M = Q
#A = schur
#wikipedia the resulting algorythm cgs
#joblib
#cise.ufl.edu/research/sparse/matrices - try as input
#try to set static positions to the vertices in visualization
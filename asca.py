from scipy.linalg import eigh, eigvalsh
from scipy.sparse import coo_matrix

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
import graph
import utils
import logging
import time

class Asca:
    def __init__(self, filename, iterations=1):
        self.current_graph = graph.GridGraph.from_hdf5(filename)
        self.solutions = list()
        self.iterations = iterations
        self.current_iteration = 0
    
    def solve_asca(self):
        # currently set for grid structures
        print(f"ASCA Iteration {self.current_iteration}")
        # selecting coarse vertices
        self.current_graph.select_coarse_moore_neighborhood(1)
        # creating subgraphs
        self.current_graph.create_subgraphs_all(1)
        #utils.visualize_graph(self.current_graph, name=f"Graph{self.current_iteration}")
        
        # initialize approximation matrix
        l = len(self.current_graph.coarse_vertices)
        Q = np.zeros(shape = (l, l), dtype=np.float64)
        # variable to visualize only first subgraph
        s = True
        
        for sub_graph in self.current_graph.get_subgraphs():
            print(f"Solving subgraph {sub_graph.name} with {len(sub_graph.vertex_list)} vertices")
            if s:
                utils.visualize_graph(sub_graph, name=f"Subgraph{self.current_iteration}")
                s = False

            mapping = sub_graph.local_to_global_mapping()
            schur_complement = sub_graph.local_schur_complement()
            temp = mapping @ schur_complement @ mapping.T
            Q += temp
        
        self.solutions.append(Q)
        
        schur = self.current_graph.local_schur_complement()
        main_graph_adj_matrix = self.current_graph.vertex_list_to_adj_matrix(self.current_graph.vertex_list)

        with pd.HDFStore("data/analysis.hdf5", mode="a") as store:
            store.put(f"analysis/iteration{self.current_iteration}/adj_matrix", pd.DataFrame(main_graph_adj_matrix))
            store.put(f"analysis/iteration{self.current_iteration}/asca", pd.DataFrame(Q))
            store.put(f"analysis/iteration{self.current_iteration}/schur_complement", pd.DataFrame(schur))
            schur = schur + 1   
            Q = Q + 1
            try:
                store.put(f"analysis/iteration{self.current_iteration}/eigen_vals", pd.DataFrame(eigh(schur, Q)[0]))
            except Exception as e:
                print("Eigh error:", e)
        
        # getting adj matrix out of laplacian matrix
        Q = abs(Q - 1)
        indexes = (range(len(Q)), range(len(Q)))
        Q[indexes] = 0
        self.current_graph = graph.GridGraph.from_coo(coo_matrix(Q))
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
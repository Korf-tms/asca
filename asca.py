from scipy.linalg import eigh, eigvalsh

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
import graph
import utils
import logging

if len(sys.argv) != 4:
    print("Usage: python asca.py <path_to_file> <rows> <cols>")
    sys.exit(1)

utils.clear_folder_or_create("data")
utils.clear_folder_or_create("images")

shape = (int(sys.argv[2]), int(sys.argv[3]))
main_graph = graph.GridGraph.from_hdf5(path=sys.argv[1], shape=shape)
main_graph.select_coarse_spacing(1)
main_graph.create_subgraphs_max(1)
utils.visualize_graph(main_graph)

Q = 0
s = True
for sub_graph in main_graph.get_subgraphs():
    if s:
        utils.visualize_graph(sub_graph)
        s = False
    mapping = sub_graph.local_to_global_mapping().toarray()
    schur_complement = sub_graph.local_schur_complement()
    temp = mapping @ schur_complement @ mapping.T
    Q += temp

schur = main_graph.local_schur_complement()
schur_arr = np.array(schur, copy=True)
q_arr = np.array(Q, copy=True)
main_graph_adj_matrix = main_graph.vertex_list_to_adj_matrix(main_graph.vertex_list)

with pd.HDFStore("data/analysis.hdf5", mode="w") as store:
    store.put("analysis/adj_matrix", pd.DataFrame(main_graph_adj_matrix))
    store.put("analysis/asca", pd.DataFrame(q_arr))
    store.put("analysis/schur_complement", pd.DataFrame(schur_arr))
    try:
        store.put("analysis/eigen_vals", pd.DataFrame(eigh(schur_arr, q_arr)[0]))
    except Exception as e:
        print("Eigh error:", e)
    try:
        store.put("analysis/eigen_vals_asca", pd.DataFrame(eigvalsh(q_arr)))
    except Exception as e:
        print("Eigvalsh error:", e)
    try:
        store.put("analysis/eigen_vals_schur", pd.DataFrame(np.linalg.eig(schur_arr)[0]))
    except Exception as e:
        print("Eigvalsh error:", e)

#psat
#popsat funkce
#parametrizace
#rekurentni schema
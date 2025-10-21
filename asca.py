from scipy.linalg import eigh

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
import graph
import utils
import logging

#logger = logging.getLogger(__name__)

#logging.basicConfig(level=logging.INFO, filename=f"log/{time.strftime("%d_%m_%Y_%M_%S")}.log", format='%(asctime)s, %(levelname)s: %(message)s')
shape = (11,11)
#utils.generate_graph_to_coo_csv(shape[0], shape[1], sys.argv[1], connection_prob=1)

main_graph = graph.GridGraph.from_hdf5(path=sys.argv[1], shape=shape)
utils.clear_folder_or_create("data")
utils.clear_folder_or_create("images")
main_graph_adj_matrix = main_graph.vertex_list_to_adj_matrix(main_graph.vertex_list)

main_graph.select_coarse_spacing(1)
main_graph.create_subgraphs_max(2)

utils.visualize_graph(main_graph)

Q = 0
for sub_graph in main_graph.get_subgraphs():
    mapping = sub_graph.local_to_global_mapping().toarray()
    schur_complement = sub_graph.local_schur_complement()
    temp = mapping @ schur_complement @ mapping.T
    Q += temp

schur = main_graph.local_schur_complement()
schur_arr = np.array(schur, copy=True)
q_arr = np.array(Q, copy=True)

eigen_val_vec = eigh(q_arr, schur_arr)

with pd.HDFStore("data/analysis.hdf5", mode="w") as store:
    store.put("analysis/adj_matrix", pd.DataFrame(main_graph_adj_matrix))
    store.put("analysis/asca", pd.DataFrame(q_arr))
    store.put("analysis/schur_complement", pd.DataFrame(schur_arr))
    store.put("analysis/eigen_vals", pd.DataFrame(eigen_val_vec[0]))

#psat
#popsat funkce
#parametrizace
#rekurentni schema
#wieghted graph
#pandas
#hdf5
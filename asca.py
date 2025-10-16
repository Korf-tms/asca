import sys
import graph
import utils
import numpy as np
import logging
import time
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh
n=10

#logger = logging.getLogger(__name__)

#logging.basicConfig(level=logging.INFO, filename=f"log/{time.strftime("%d_%m_%Y_%M_%S")}.log", format='%(asctime)s, %(levelname)s: %(message)s')

#utils.generate_graph_to_coo_csv(5, 5, sys.argv[1])
utils.clear_folder("csv")
utils.clear_folder("images")

shape = (5,5)

main_graph = graph.GridGraph.from_csv(path=sys.argv[1], shape=shape)

main_graph_adj_matrix = main_graph.vertex_list_to_adj_matrix(main_graph.vertex_list)
np.savetxt("csv/adj_matrix.csv", main_graph_adj_matrix, delimiter=",", fmt="%.2f")

main_graph.select_coarse_spacing()
main_graph.create_subgraphs_max()

utils.visualize_graph(main_graph)

Q = 0
for sub_graph in main_graph.get_subgraphs():
    utils.visualize_graph(sub_graph)
    mapping = sub_graph.local_to_global_mapping().toarray()
    schur_complement = sub_graph.local_schur_complement()
    temp = mapping @ schur_complement @ mapping.T
    Q += temp

schur = main_graph.local_schur_complement()

np.savetxt("csv/asca.csv", Q, delimiter=",", fmt="%.2f") 
np.savetxt("csv/schurs_complement.csv", schur, delimiter=",", fmt="%.2f") 
np.savetxt("csv/eig_vals.csv", eigh(Q, schur)[0] , delimiter=",", fmt="%.2f") 

#parametrizace
#rekurentni schema
#wieghted graph
#pandas
#hdf5
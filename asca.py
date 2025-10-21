from scipy.linalg import eigh

import matplotlib.pyplot as plt
import numpy as np

import sys
import graph
import utils
import logging

#logger = logging.getLogger(__name__)

#logging.basicConfig(level=logging.INFO, filename=f"log/{time.strftime("%d_%m_%Y_%M_%S")}.log", format='%(asctime)s, %(levelname)s: %(message)s')
shape = (11,11)
utils.generate_graph_to_coo_csv(shape[0], shape[1], sys.argv[1], connection_prob=1)
utils.clear_folder("csv")
utils.clear_folder("images")

main_graph = graph.GridGraph.from_csv(path=sys.argv[1], shape=shape)

main_graph_adj_matrix = main_graph.vertex_list_to_adj_matrix(main_graph.vertex_list)
np.savetxt("csv/adj_matrix.csv", main_graph_adj_matrix, delimiter=",", fmt="%.2f")

main_graph.select_coarse_spacing(1)
main_graph.create_subgraphs_max(3)

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

fig, axis = plt.subplots(1, 3)

axis[0].imshow(main_graph_adj_matrix, cmap="Greys", interpolation='none', vmin=0, vmax=1)
axis[0].set_title("Adj Matrix")

axis[1].imshow(np.abs(q_arr), cmap="Greys", interpolation='none', vmin=0, vmax=schur_arr.max())
axis[1].set_title("ASCA")

axis[2].imshow(np.abs(schur_arr), cmap="Greys", interpolation='none', vmin=0, vmax=schur_arr.max())
axis[2].set_title("Schur")

plt.tight_layout()
plt.show()

np.savetxt("csv/asca.csv", q_arr, delimiter=",", fmt="%f") 
np.savetxt("csv/schurs_complement.csv", schur_arr, delimiter=",", fmt="%f") 
np.savetxt("csv/eigen_val.csv", eigh(q_arr, schur_arr)[0], delimiter=",", fmt="%f")



#psat
#popsat funkce
#parametrizace
#rekurentni schema
#wieghted graph
#pandas
#hdf5
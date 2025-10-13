import sys
import graph
import utils
import numpy as np
import logging
import time
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh
n=10

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, filename=f"log/{time.strftime("%d_%m_%Y_%M_%S")}.log", format='%(asctime)s, %(levelname)s: %(message)s')

utils.generate_graph_to_coo_csv(5, 5, sys.argv[1])

main_graph = graph.Graph(path=sys.argv[1])
utils.visualize_graph(main_graph)
coarse_graph = graph.CoarseGraph(main_graph.coarse_vertices, main_graph)
utils.visualize_graph(coarse_graph)
Q = 0
for vertex in coarse_graph.vertex_list:
    utils.visualize_graph(vertex.graph)
    mapping = vertex.graph.local_to_global_mapping().toarray()
    schur_complement = vertex.graph.local_schur_complement()
    temp = mapping @ schur_complement @ mapping.T
    Q += temp

schur = main_graph.local_schur_complement()
np.savetxt("csv/asca.csv", Q, delimiter=",", fmt="%.2f") 
np.savetxt("csv/sc.csv", schur, delimiter=",", fmt="%.2f") 
np.savetxt("csv/test.csv", eigh(Q, schur)[0] , delimiter=",", fmt="%.2f") 

#parametrizace
#rekurentni schema
#wieghted graph
#pandas
#mrizky 
#input graf, rozdeleni, poddomeny
#hdf5
#redo class logic
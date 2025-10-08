import graph
import utils
import numpy as np
import logging
import time
n=10

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, filename=f"log/{time.strftime("%d_%m_%Y_%M_%S")}.log", format='%(asctime)s, %(levelname)s: %(message)s')

utils.generate_graph_to_coo_csv(10, 10, "input.csv")

main_graph = graph.Graph()
main_graph.init_from_csv("C:/Other/School/ZaverecnaPrace/input.csv")
mis = main_graph.maximal_independent_set()
utils.visualize_graph(main_graph)
coarse_graph = graph.CoarseGraph(mis, main_graph)
utils.visualize_graph(coarse_graph)
Q = 0
for vertex in coarse_graph.vertex_list:
    utils.visualize_graph(vertex.graph)
    mapping = vertex.graph.local_to_global_mapping().toarray()
    schur_complement = vertex.graph.local_schur_complement()
    temp = mapping @ schur_complement @ mapping.T
    logging.info(f"mapping * schur complement * mapping.T:\n{temp}")
    Q += temp
    logging.info(f"current Q:\n{Q}")

logging.info(f"count matrix:\n{coarse_graph.count_matrix}")
schur = main_graph.local_schur_complement()
np.savetxt("csv/asca.csv", Q, delimiter=",", fmt="%.2f") 
np.savetxt("csv/sc.csv", schur, delimiter=",", fmt="%.2f") 

#multiplicity
#parametrizace
#vlastni cisla, cim bliz k nule, eigen value
#rekurentni schema
#wieghted graph
#pandas
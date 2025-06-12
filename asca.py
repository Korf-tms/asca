import graph
import utils
import networkx as nx
import numpy as np

original_graph = graph.OriginalGraph("graph100x100.json")
coarse_graph = graph.CoarseGraph(original_graph.maximal_independent_set(), original_graph)
Q = 0
for vertex in coarse_graph.vertex_dict.values():
    mapping = vertex.graph.local_to_global_mapping().toarray()
    schur_complement = vertex.graph.local_schur_complement()
    Q += mapping @ schur_complement @ mapping.T
print(Q)
np.savetxt("output.txt", Q, fmt='%s')
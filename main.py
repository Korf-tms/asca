import asca
import evaluation
'''
# example of running asca
a = asca.Asca(
    filename="matrices/11x11.hdf5",
    output_file="data/11x11.hdf5",
    iterations=1,
    coarse_selection_method="moore",
    coarse_selection_method_arguments={"size": 1},
    create_subgraphs_method="moore_coarse",
    create_subgraphs_method_arguments={"size": 2},
)
a.run_approximation()
'''

e = evaluation.Evaluator("data/11x11.hdf5")
e.cg_evaluation()
e.eigsh_evaluation()


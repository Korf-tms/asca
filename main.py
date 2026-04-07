import asca
import evaluation

# example of running asca
a = asca.Asca(
    filename="matrices/5x5grid.hdf5",
    output_file="data/5x5grid.hdf5",
    iterations=1,
    coarse_selection_method="mis",
    coarse_selection_method_arguments={"size": 1},
    create_subgraphs_method="depth",
    create_subgraphs_method_arguments={"size": 2},
)
a.run_approximation()

e = evaluation.Evaluator("data/5x5grid.hdf5")
e.cgs_evaluation()
e.eigsh_evaluation()

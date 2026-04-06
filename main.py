import asca
import evaluation
#example of running asca
a = asca.Asca(
    filename="matrices/dwt_607.mat",
    output_file="data/dwt.hdf5",
    iterations=1,
    coarse_selection_method="mis",
    coarse_selection_method_arguments={"size":1},
    create_subgraphs_method="depth",
    create_subgraphs_method_arguments={"max_depth":2}
)
a.run_approximation()

e = evaluation.Evaluator("data/dwt.hdf5")
e.cgs_evaluation()
e.eigsh_evaluation()
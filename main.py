import asca
import evaluation
#example of running asca
a = asca.Asca(
    filename="matrices/11x11.hdf5",
    output_file="data/test.hdf5",
    iterations=1,
    coarse_selection_method="moore",
    coarse_selection_method_arguments={"size":1},
    create_subgraphs_method="moore_coarse",
    create_subgraphs_method_arguments={"size":2}
)
a.run_approximation()

e = evaluation.Evaluator("data/test.hdf5")
print(e.cgs_evaluation())
print(e.eigsh_evaluation()[0])
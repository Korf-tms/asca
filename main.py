import asca
#example of running asca
a = asca.Asca(
    filename="11x11.hdf5",
    iterations=1,
    coarse_selection_method="moore",
    coarse_selection_method_arguments={"size":1},
    create_subgraphs_method="moore_coarse",
    create_subgraphs_method_arguments={"size":2}
)
a.run_approximation()
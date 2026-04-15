import asca
import evaluation
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    filename="log.log",
)

# example of running asca

filename = "bcsstk05"
tests = [
    ("mis", {"size" : 1}, "depth", {"size" : 4}),
    ("mis", {"size" : 1}, "depth", {"size" : 6}),
    ("mis_min", {"size" : 1}, "depth", {"size" : 4}),
    ("mis_min", {"size" : 1}, "depth", {"size" : 6}),
    ("mis_max", {"size" : 1}, "depth", {"size" : 4}),
    ("mis_max", {"size" : 1}, "depth", {"size" : 6})
]

for coarse_method, cm_args, subgraph_method, sm_args in tests:
    output_file = f"data/{filename}_{coarse_method}{cm_args["size"]}_{subgraph_method}{sm_args["size"]}.hdf5"
    
    a = asca.Asca(
        filename=f"matrices/{filename}.mtx.gz",
        output_file=output_file,
        iterations=1,
        coarse_selection_method=coarse_method,
        coarse_selection_method_arguments=cm_args,
        create_subgraphs_method=subgraph_method,
        create_subgraphs_method_arguments=sm_args,
    )
    a.run_approximation()

    e = evaluation.Evaluator(output_file)
    e.cg_evaluation()
    e.eigsh_evaluation()


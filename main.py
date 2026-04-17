import pathlib as pl

import asca
import evaluation
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    filename="log.log",
)

# example of running asca
cs_methods = []
cs_arguments = []
sc_methods = []
sc_arguments = []

for cs_method in ["moore"]:
    for sc_method in ["moore_coarse"]:
        for cs_argument in [0, 1, 2, 3, 4]:
            for sc_argument in [2, 4, 10]:
                cs_methods.append(cs_method)
                cs_arguments.append({"size":cs_argument+1})
                sc_methods.append(sc_method)
                sc_arguments.append({"size":sc_argument+cs_argument})

config = asca.AscaConfig(
    filename="matrices/110x110.hdf5",
    coarse_selection_method=cs_methods,
    coarse_selection_method_arguments=cs_arguments,
    subgraph_creation_method=sc_methods,
    subgraph_creation_method_arguments=sc_arguments,
    output_file=f"data/110x110.hdf5",
    iterations=1
)

asca.run_approximation(config)

folder = pl.Path("data")
files = [p.name for p in folder.iterdir() if p.is_file()]

for file in files:
    e = evaluation.Evaluator(f"data/{file}")
    e.cg_evaluation()
    e.eigsh_evaluation()

import logging

import pathlib as pl

from asca import AscaConfig, EvaluatorConfig, run_evaluation, run_approximation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    filename="log.log",
)

# example of running asca

grid_config = AscaConfig(
    filename="matrices/110x110.hdf5",
    coarse_selection_method=["moore"] * 3,
    coarse_selection_method_arguments=[{"size": 1}] * 3,
    subgraph_creation_method=["moore_coarse"] * 3,
    subgraph_creation_method_arguments=[{"size": 2}, {"size": 4}, {"size": 10}],
    output_file="data/110x110.hdf5",
    iterations=1,
)

run_approximation(grid_config)

skirt_config1 = AscaConfig(
    filename="matrices/skirt.mtx",
    coarse_selection_method=["mis"] * 3,
    coarse_selection_method_arguments=[{"size": 1}] * 3,
    subgraph_creation_method=["depth"] * 3,
    subgraph_creation_method_arguments=[{"size": 4}, {"size": 6}, {"size": 10}],
    output_file="data/skirt.hdf5",
    iterations=1,
)

run_approximation(skirt_config1)

skirt_config2 = AscaConfig(
    filename="matrices/skirt.mtx",
    coarse_selection_method=["mis_strength_desc"] * 3,
    coarse_selection_method_arguments=[{"size": 1}] * 3,
    subgraph_creation_method=["depth"] * 3,
    subgraph_creation_method_arguments=[{"size": 4}, {"size": 6}, {"size": 10}],
    output_file="data/skirt.hdf5",
    iterations=1,
)

run_approximation(skirt_config2)

annulus_config = AscaConfig(
    filename="matrices/annulus.mtx",
    coarse_selection_method=["mis"] * 3,
    coarse_selection_method_arguments=[{"size": 1}] * 3,
    subgraph_creation_method=["depth"] * 3,
    subgraph_creation_method_arguments=[{"size": 2}, {"size": 4}, {"size": 10}],
    output_file="data/annulus.hdf5",
    iterations=1,
)
"""
run_approximation(annulus_config)

olafu_config1 = AscaConfig(
    filename="matrices/olafu.mtx",
    coarse_selection_method=["mis_strength_asc"] * 4,
    coarse_selection_method_arguments=[{"size":1}] * 4,
    subgraph_creation_method=["macrostructure"] * 4,
    subgraph_creation_method_arguments=[
        {"micro_size":3, "connection_depth":2, "merge_distance":1},
        {"micro_size":4, "connection_depth":2, "merge_distance":1},
        {"micro_size":5, "connection_depth":2, "merge_distance":1},
        {"micro_size":6, "connection_depth":2, "merge_distance":1}
        ],
    output_file="data/olafu.hdf5",
    iterations=1
)

run_approximation(olafu_config1)

olafu_config2 = AscaConfig(
    filename="matrices/olafu.mtx",
    coarse_selection_method=["mis_strength_asc"] * 4,
    coarse_selection_method_arguments=[{"size":1}] * 4,
    subgraph_creation_method=["depth"] * 4,
    subgraph_creation_method_arguments=[{"size":3}, {"size":4}, {"size":5}, {"size":6}],
    output_file="data/olafu.hdf5",
    iterations=1
)

run_approximation(olafu_config2)

ct2010_config1 = AscaConfig(
    filename="matrices/ct2010.mtx",
    coarse_selection_method=["mis_strength_asc"] * 4,
    coarse_selection_method_arguments=[{"size":1}] * 4,
    subgraph_creation_method=["depth"] * 4,
    subgraph_creation_method_arguments=[{"size":3}, {"size":4}, {"size":5}, {"size":6}],
    output_file="data/ct2010.hdf5",
    iterations=1
)

run_approximation(ct2010_config1)

ct2010_config2 = AscaConfig(
    filename="matrices/ct2010.mtx",
    coarse_selection_method=["mis_strength_asc"] * 4,
    coarse_selection_method_arguments=[{"size":1}] * 4,
    subgraph_creation_method=["macrostructure"] * 4,
    subgraph_creation_method_arguments=[
        {"micro_size":3, "connection_depth":2, "merge_distance":1},
        {"micro_size":4, "connection_depth":2, "merge_distance":1},
        {"micro_size":5, "connection_depth":2, "merge_distance":1},
        {"micro_size":6, "connection_depth":2, "merge_distance":1}
        ],
    output_file="data/ct2010.hdf5",
    iterations=1
)

run_approximation(ct2010_config2)
"""
# evaluates all results

folder = pl.Path("data")
files = [
    f"data/{p.name}"
    for p in folder.iterdir()
    if p.is_file() and "evaluation" not in p.stem
]

evauation_config = EvaluatorConfig(input_files=files)

run_evaluation(evauation_config)

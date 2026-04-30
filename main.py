import logging
from dataclasses import dataclass

import pathlib as pl
import matplotlib.pyplot as plt

from asca import AscaConfig, EvaluatorConfig, run_evaluation, run_approximation
from asca.visualization import (
    cg_error_history,
    cg_residual_history,
    eigenvalues,
    approximation,
)

# This script creates the data and figures used in the work.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    filename="log.log",
)

# Run ASCA on all selected graphs.

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

run_approximation(annulus_config)

olafu_config1 = AscaConfig(
    filename="matrices/olafu.mtx",
    coarse_selection_method=["mis_strength_asc"] * 3,
    coarse_selection_method_arguments=[{"size": 1}] * 3,
    subgraph_creation_method=["macrostructure"] * 3,
    subgraph_creation_method_arguments=[
        {"micro_size": 3, "connection_depth": 2, "merge_distance": 1},
        {"micro_size": 4, "connection_depth": 2, "merge_distance": 1},
        {"micro_size": 5, "connection_depth": 2, "merge_distance": 1},
    ],
    output_file="data/olafu.hdf5",
    iterations=1,
)

run_approximation(olafu_config1)

olafu_config2 = AscaConfig(
    filename="matrices/olafu.mtx",
    coarse_selection_method=["mis_strength_asc"] * 3,
    coarse_selection_method_arguments=[{"size": 1}] * 3,
    subgraph_creation_method=["depth"] * 3,
    subgraph_creation_method_arguments=[{"size": 3}, {"size": 4}, {"size": 5}],
    output_file="data/olafu.hdf5",
    iterations=1,
)

run_approximation(olafu_config2)

# This graph is quite large and may take much longer to finish than the others.

ct2010_config1 = AscaConfig(
    filename="matrices/ct2010.mtx",
    coarse_selection_method=["mis_strength_asc"] * 3,
    coarse_selection_method_arguments=[{"size": 1}] * 3,
    subgraph_creation_method=["depth"] * 3,
    subgraph_creation_method_arguments=[{"size": 3}, {"size": 4}, {"size": 5}],
    output_file="data/ct2010.hdf5",
    iterations=1,
)

run_approximation(ct2010_config1)

ct2010_config2 = AscaConfig(
    filename="matrices/ct2010.mtx",
    coarse_selection_method=["mis_strength_asc"] * 3,
    coarse_selection_method_arguments=[{"size": 1}] * 3,
    subgraph_creation_method=["macrostructure"] * 3,
    subgraph_creation_method_arguments=[
        {"micro_size": 3, "connection_depth": 2, "merge_distance": 1},
        {"micro_size": 4, "connection_depth": 2, "merge_distance": 1},
        {"micro_size": 5, "connection_depth": 2, "merge_distance": 1},
    ],
    output_file="data/ct2010.hdf5",
    iterations=1,
)

run_approximation(ct2010_config2)

# Evaluate selected results.


folder = pl.Path("data")
files = [
    f"data/{p.name}"
    for p in folder.iterdir()
    if p.is_file()
    and "evaluation" not in p.stem
    and "annulus" not in p.stem
    and "110x110" not in p.stem
]

evauation_config = EvaluatorConfig(input_files=files)

run_evaluation(evauation_config)

# Create CG error, residual history, and eigenvalue figures.


@dataclass
class Group:
    plots: list[list[tuple[int, pl.Path]]]
    output_name: str
    plot_history: bool
    plot_eig: bool


def test_file(number: int, filename: str) -> tuple[int, pl.Path]:
    return number, pl.Path("evaluation") / filename


groups: list[Group] = []

groups.append(
    Group(
        [
            [
                test_file(1, "110x110_moore_size1_moore_coarse_size2_evaluation.hdf5"),
                test_file(2, "110x110_moore_size1_moore_coarse_size4_evaluation.hdf5"),
                test_file(3, "110x110_moore_size1_moore_coarse_size10_evaluation.hdf5"),
            ],
        ],
        "grid",
        False,
        True,
    )
)

groups.append(
    Group(
        [
            [
                test_file(4, "skirt_mis_size1_depth_size4_evaluation.hdf5"),
                test_file(5, "skirt_mis_size1_depth_size6_evaluation.hdf5"),
                test_file(6, "skirt_mis_size1_depth_size10_evaluation.hdf5"),
            ],
            [
                test_file(
                    7, "skirt_mis_strength_desc_size1_depth_size4_evaluation.hdf5"
                ),
                test_file(
                    8, "skirt_mis_strength_desc_size1_depth_size6_evaluation.hdf5"
                ),
                test_file(
                    9, "skirt_mis_strength_desc_size1_depth_size10_evaluation.hdf5"
                ),
            ],
        ],
        "skirt",
        True,
        True,
    )
)

groups.append(
    Group(
        [
            [
                test_file(10, "annulus_mis_size1_depth_size2_evaluation.hdf5"),
                test_file(11, "annulus_mis_size1_depth_size4_evaluation.hdf5"),
                test_file(12, "annulus_mis_size1_depth_size10_evaluation.hdf5"),
            ],
        ],
        "annulus",
        True,
        True,
    )
)

groups.append(
    Group(
        [
            [
                test_file(
                    13,
                    "olafu_mis_strength_asc_size1_macrostructure_connection_depth2_merge_distance1_micro_size3_evaluation.hdf5",
                ),
                test_file(
                    14,
                    "olafu_mis_strength_asc_size1_macrostructure_connection_depth2_merge_distance1_micro_size4_evaluation.hdf5",
                ),
                test_file(
                    15,
                    "olafu_mis_strength_asc_size1_macrostructure_connection_depth2_merge_distance1_micro_size5_evaluation.hdf5",
                ),
            ],
            [
                test_file(
                    16, "olafu_mis_strength_asc_size1_depth_size3_evaluation.hdf5"
                ),
                test_file(
                    17, "olafu_mis_strength_asc_size1_depth_size4_evaluation.hdf5"
                ),
                test_file(
                    18, "olafu_mis_strength_asc_size1_depth_size5_evaluation.hdf5"
                ),
            ],
        ],
        "olafu",
        True,
        True,
    )
)

groups.append(
    Group(
        [
            [
                test_file(
                    19,
                    "ct2010_mis_strength_asc_size1_macrostructure_connection_depth2_merge_distance1_micro_size3_evaluation.hdf5",
                ),
                test_file(
                    20,
                    "ct2010_mis_strength_asc_size1_macrostructure_connection_depth2_merge_distance1_micro_size4_evaluation.hdf5",
                ),
                test_file(
                    21,
                    "ct2010_mis_strength_asc_size1_macrostructure_connection_depth2_merge_distance1_micro_size5_evaluation.hdf5",
                ),
            ],
            [
                test_file(
                    22, "ct2010_mis_strength_asc_size1_depth_size3_evaluation.hdf5"
                ),
                test_file(
                    23, "ct2010_mis_strength_asc_size1_depth_size4_evaluation.hdf5"
                ),
                test_file(
                    24, "ct2010_mis_strength_asc_size1_depth_size5_evaluation.hdf5"
                ),
            ],
        ],
        "ct2010",
        True,
        False,
    )
)


for group in groups:
    size = len(group.plots)
    if group.plot_eig:
        eig_fig, eig_plot = plt.subplots(
            nrows=1, ncols=size, figsize=(size * 8, 5), squeeze=False
        )
        for i, files in enumerate(group.plots):
            eigenvalues(
                files, colors=["darkred", "red", "lightcoral"], ax=eig_plot[0, i]
            )
        eig_fig.tight_layout()
        eig_fig.savefig(f"figures/{group.output_name}_eigenvalues.pdf", dpi=500)
    if group.plot_history:
        for i, files in enumerate(group.plots):
            history_fig, history_plot = plt.subplots(
                nrows=1, ncols=2, figsize=(10, 5), squeeze=False
            )
            cg_error_history(
                files,
                ax=history_plot[0, 0],
                colors=["darkred", "red", "lightcoral"],
            )
            cg_residual_history(
                files, colors=["darkblue", "blue", "lightblue"], ax=history_plot[0, 1]
            )
            history_fig.tight_layout()
            history_fig.savefig(f"figures/{group.output_name}_history{i}.pdf", dpi=500)

# Create approximation matrix heatmap figures.


@dataclass
class Group:
    files: list[str]
    name: str


file_groups = [
    Group(
        [
            "data/110x110_moore_size1_moore_coarse_size2.hdf5",
            "data/110x110_moore_size1_moore_coarse_size10.hdf5",
        ],
        "grid",
    ),
    Group(
        [
            "data/skirt_mis_strength_desc_size1_depth_size4.hdf5",
            "data/skirt_mis_strength_desc_size1_depth_size10.hdf5",
        ],
        "skirt",
    ),
    Group(
        [
            "data/annulus_mis_size1_depth_size2.hdf5",
            "data/annulus_mis_size1_depth_size10.hdf5",
        ],
        "annulus",
    ),
    Group(
        [
            "data/olafu_mis_strength_asc_size1_depth_size4.hdf5",
            "data/olafu_mis_strength_asc_size1_macrostructure_connection_depth2_merge_distance1_micro_size4.hdf5",
        ],
        "olafu",
    ),
    Group(
        [
            "data/ct2010_mis_strength_asc_size1_depth_size4.hdf5",
            "data/ct2010_mis_strength_asc_size1_macrostructure_connection_depth2_merge_distance1_micro_size4.hdf5",
        ],
        "ct2010",
    ),
]

for group in file_groups:
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(2 * 5, 5), squeeze=False)
    for i, file in enumerate(group.files):
        approximation(file=file, ax=ax[0, i])
    fig.tight_layout()
    fig.savefig(f"figures/{group.name}_heatmap.png", dpi=500)

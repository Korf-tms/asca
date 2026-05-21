import logging
import sys
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
    handlers=[
        logging.FileHandler("log.log"),
        logging.StreamHandler(sys.stdout)
    ]
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

# Evaluate selected results.


folder = pl.Path("data")
files = [
    f"data/{p.name}"
    for p in folder.iterdir()
    if p.is_file()
    and "evaluation" not in p.stem
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
        True,
        True,
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
                files, colors=["darkred", "red", "lightcoral"], ax=eig_plot[0, i], plot_type="lines"
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
            "data/110x110_moore_size1_moore_coarse_size4.hdf5",
            "data/110x110_moore_size1_moore_coarse_size10.hdf5",
        ],
        "grid",
    ),
]

for group in file_groups:
    ncols = len(group.files)
    fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 5, 5), squeeze=False)
    for i, file in enumerate(group.files):
        file_stem = pl.Path(file).stem
        approximation(
            file=file,
            ax=ax[0, i],
            title=f"{file_stem}",
        )
    fig.tight_layout()
    fig.savefig(f"figures/{group.name}_sparsity.png", dpi=600, bbox_inches="tight")

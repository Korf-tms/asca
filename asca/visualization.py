import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np
import pathlib as pl
import h5py

from . import utils


def _is_numbered_file(file):
    return (
        isinstance(file, tuple)
        and len(file) == 2
        and isinstance(file[0], int)
        and isinstance(file[1], (str, pl.Path))
    )


def _normalize_files(file):
    if isinstance(file, (str, pl.Path)) or _is_numbered_file(file):
        files = [file]
    else:
        files = list(file)

    numbered_files = []
    for current_file in files:
        if _is_numbered_file(current_file):
            numbered_files.append(current_file)
        else:
            numbered_files.append((None, current_file))

    return numbered_files


def _test_label(number, fallback):
    if number is None:
        return str(fallback)
    return f"Test number {number}"


def _cg_common(
    file: (
        list[str]
        | list[pl.Path]
        | list[tuple[int, str | pl.Path]]
        | str
        | pl.Path
        | tuple[int, str | pl.Path]
    ),
    title: str,
    y_label: str,
    x_label: str,
    name: str,
    iteration: int = 0,
    labels: list[str] | None = None,
    colors: list[str] | None = None,
    linestyle: list[str] | None = None,
    ax=None,
    output_file: str | None = None,
):
    histories = []

    for i, (number, current_file) in enumerate(_normalize_files(file)):
        with h5py.File(current_file) as hdf_file:
            histories.append(
                {
                    "number": number,
                    "history": hdf_file[f"iteration{iteration}/{name}"][:],
                    "label": (
                        labels[i] if labels is not None else _test_label(number, i + 1)
                    ),
                    "color": None if colors is None else colors[i],
                    "linestyle": None if linestyle is None else linestyle[i],
                }
            )

    if len(histories) == 0:
        raise ValueError("No files were provided for plotting.")

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        created_fig = True

    if any(history["number"] is not None for history in histories):
        histories.sort(
            key=lambda history: (
                history["number"] if history["number"] is not None else float("inf")
            )
        )
    else:
        histories.sort(key=lambda history: len(history["history"]))

    for history in histories:
        ax.plot(
            range(len(history["history"])),
            history["history"],
            marker="o",
            linestyle="-" if history["linestyle"] is None else history["linestyle"],
            color="lightcoral" if history["color"] is None else history["color"],
            label=history["label"],
        )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.set_yscale("log")

    max_history_len = max([len(x["history"]) for x in histories])
    x_vals = range(0, max_history_len + 1)
    ax.set_xticks(x_vals)
    ax.set_xticklabels(range(1, max_history_len + 2))

    ax.yaxis.set_minor_locator(LogLocator(subs=range(2, 10), numticks=100))
    ax.yaxis.set_minor_formatter(NullFormatter())

    ax.grid(True, which="major", axis="y", linestyle="--", linewidth=0.5)
    ax.grid(True, which="minor", axis="y", linestyle=":", linewidth=0.3)
    ax.grid(True, which="major", axis="x", linestyle="--", linewidth=0.5)

    ax.legend()

    if created_fig:
        plt.tight_layout()
        if output_file is not None:
            plt.savefig(output_file, dpi=500)
        plt.show()

    return ax


def cg_error_history(
    file: (
        list[str]
        | list[pl.Path]
        | list[tuple[int, str | pl.Path]]
        | str
        | pl.Path
        | tuple[int, str | pl.Path]
    ),
    iteration: int = 0,
    labels: list[str] | None = None,
    colors: list[str] | None = None,
    linestyle: list[str] | None = None,
    ax=None,
    output_file: str | None = None,
):
    return _cg_common(
        file=file,
        title="CG error history",
        y_label="Relative error",
        x_label="CG iteration",
        name="error_history",
        iteration=iteration,
        labels=labels,
        colors=colors,
        linestyle=linestyle,
        ax=ax,
        output_file=output_file,
    )


def cg_residual_history(
    file: (
        list[str]
        | list[pl.Path]
        | list[tuple[int, str | pl.Path]]
        | str
        | pl.Path
        | tuple[int, str | pl.Path]
    ),
    iteration: int = 0,
    labels: list[str] | None = None,
    colors: list[str] | None = None,
    linestyle: list[str] | None = None,
    ax=None,
    output_file: str | None = None,
):

    return _cg_common(
        file=file,
        title="CG residual history",
        y_label="Relative residual",
        x_label="CG iteration",
        name="residual_history",
        iteration=iteration,
        labels=labels,
        colors=colors,
        linestyle=linestyle,
        ax=ax,
        output_file=output_file,
    )


def eigenvalues(
    file: (
        list[str]
        | list[pl.Path]
        | list[tuple[int, str | pl.Path]]
        | str
        | pl.Path
        | tuple[int, str | pl.Path]
    ),
    labels: list[str] | None = None,
    colors: list[str] | None = None,
    iteration: int = 0,
    ax=None,
    output_file: str | None = None,
    plot_type: str = "lines",
):
    eigenvalues_condition_numbers = []

    for i, (number, current_file) in enumerate(_normalize_files(file)):
        with h5py.File(current_file) as hdf_file:
            eigenvalues_condition_numbers.append(
                {
                    "number": number,
                    "eigenvalues": hdf_file[f"iteration{iteration}/eigenvalues"][:],
                    "condition_number": hdf_file[
                        f"iteration{iteration}/condition_number"
                    ][()],
                    "label": (
                        labels[i] if labels is not None else _test_label(number, i + 1)
                    ),
                    "color": None if colors is None else colors[i],
                }
            )

    if len(eigenvalues_condition_numbers) == 0:
        raise ValueError("No files were provided for plotting.")

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        created_fig = True

    spacing = 0.5

    if any(item["number"] is not None for item in eigenvalues_condition_numbers):
        eigenvalues_condition_numbers.sort(
            key=lambda item: (
                item["number"] if item["number"] is not None else float("inf")
            )
        )
    else:
        eigenvalues_condition_numbers.sort(
            key=lambda item: max(item["eigenvalues"]), reverse=True
        )
    if plot_type == "boxplot":
        bp = ax.boxplot(
            x=[x["eigenvalues"] for x in eigenvalues_condition_numbers],
            positions=[i * spacing for i in range(len(eigenvalues_condition_numbers))],
            tick_labels=[x["label"] for x in eigenvalues_condition_numbers],
            medianprops=dict(color="black"),
            patch_artist=True,
            boxprops=dict(facecolor="lightblue"),
            widths=0.2,
        )
        for box, item in zip(bp["boxes"], eigenvalues_condition_numbers):
            if item["color"] is not None:
                color = item["color"]
                box.set_facecolor(color)
                box.set_edgecolor("black")
        for i, eigenvalue_condition_number in enumerate(eigenvalues_condition_numbers):
            ax.hlines(
                y=eigenvalue_condition_number["condition_number"],
                xmin=i * spacing - 0.1,
                xmax=i * spacing + 0.1,
                colors="black",
                linewidth=2,
                label="Condition number" if i == 0 else "",
                linestyles="dashed",
            )
    elif plot_type == "curves":
        for item in eigenvalues_condition_numbers:
            ax.plot(
                item["eigenvalues"],
                marker="o",
                linestyle="-",
                color="lightcoral" if item["color"] is None else item["color"],
                label=item["label"],
            )
    elif plot_type == "lines":
        for i, item in enumerate(eigenvalues_condition_numbers, start=1):
            ax.plot(
                item["eigenvalues"],
                [i for _ in item["eigenvalues"]],
                marker="o",
                linestyle="-",
                color="lightcoral" if item["color"] is None else item["color"],
                label=item["label"],
            )

        y_ticks = list(range(1, len(eigenvalues_condition_numbers) + 1))
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([item["label"] for item in eigenvalues_condition_numbers])
        ax.set_ylim(0.5, len(eigenvalues_condition_numbers) + 0.5)

    else:
        raise ValueError(f"Invalid plot type: {plot_type}")

    ax.set_title("Eigenvalues")
    if plot_type != "lines":
        ax.set_xlabel("")
        ax.set_ylabel("Eigenvalues")
        ax.grid(True, which="major", axis="y", linestyle="--", linewidth=0.5)
        ax.grid(True, which="minor", axis="y", linestyle=":", linewidth=0.3)
        ax.grid(True, which="major", axis="x", linestyle="--", linewidth=0.5)
    else:
        ax.set_xlabel("Eigenvalues")

    ax.legend()

    if created_fig:
        plt.tight_layout()
        if output_file is not None:
            plt.savefig(output_file, dpi=500)
        plt.show()

    return ax


def approximation(
    file: pl.Path | str,
    iteration: int = 0,
    ax=None,
    output_file: str | None = None,
    title: str | None = None,
):
    matrix = 0
    with h5py.File(file) as hdf_file:
        matrix = utils.read_csr_matrix(hdf_file[f"iteration{iteration}/approximation"])

    binary = matrix.copy()
    binary.data = np.ones_like(binary.data, dtype=int)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
        created_fig = True

    ax.spy(matrix, markersize=0.1, color="black")
    ax.set_title(f"Approximation sparsity pattern (iteration {iteration})")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")

    if created_fig:
        plt.tight_layout()
        if output_file is not None:
            plt.savefig(output_file, dpi=500, box_inches="tight")
        plt.show()

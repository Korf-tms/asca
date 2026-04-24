import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import pathlib as pl
import h5py

def _cg_common(
        file : list[str] | list[pl.Path] | str | pl.Path, 
        title : str,
        y_label : str,
        x_label : str,
        name : str,
        iteration : int = 0,
        labels : list[str] | None = None, 
        colors : list[str] | None = None, 
        linestyle  : list[str] | None = None,
        ax = None,
        output_file : str | None = None
        ):
    
    

    files = file
    if isinstance(file, (str, pl.Path)):
        files = [file]

    histories = []

    for current_file in files:
        with h5py.File(current_file) as hdf_file:
            histories.append(hdf_file[f"iteration{iteration}/{name}"][:])

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        created_fig = True

    for i, history in enumerate(histories):
        ax.plot(
            range(len(history)),
            history,
            marker="o",
            linestyle="-" if linestyle is None else linestyle[i],
            color="lightcoral" if colors is None else colors[i],
            label=str(i+1) if labels is None else labels[i]
        )
    
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.set_yscale("log")

    x_vals = range(0, max([len(x) for x in histories])+1)
    ax.set_xticks(x_vals)
    ax.set_xticklabels(x_vals)

    ax.yaxis.set_minor_locator(LogLocator(subs=range(2, 10), numticks=100))
    ax.yaxis.set_minor_formatter(NullFormatter())

    ax.grid(True, which="major", axis="y", linestyle="--", linewidth=0.5)
    ax.grid(True, which="minor", axis="y", linestyle=":", linewidth=0.3)
    ax.grid(True, which="major", axis="x", linestyle="--", linewidth=0.5)
    ax.set_xlim(right=max([len(x) for x in histories])+1)

    if created_fig:
        ax.legend()
        plt.tight_layout()
        if output_file is not None:
            plt.savefig(output_file, dpi=500)
        plt.show()
    
    return ax
    
def cg_error_history(
        file : list[str] | list[pl.Path] | str | pl.Path, 
        iteration : int = 0,
        labels : list[str] | None = None, 
        colors : list[str] | None = None, 
        linestyle  : list[str] | None = None,
        ax = None,
        output_file : str | None = None
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
        output_file=output_file
    )

def cg_residual_history(
        file : list[str] | list[pl.Path] | str | pl.Path, 
        iteration : int = 0,
        labels : list[str] | None = None, 
        colors : list[str] | None = None, 
        linestyle  : list[str] | None = None,
        ax = None,
        output_file : str | None = None
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
        output_file=output_file
    )

def eigenvalues(
        file : list[str] | list[pl.Path] | str | pl.Path, 
        labels : list[str] | None = None, 
        iteration : int = 0,
        ax=None,
        output_file : str | None = None
    ):
    files = file
    if isinstance(file, (str, pl.Path)):
        files = [file]

    eigenvalues = []
    condition_numbers = []

    for current_file in files:
        with h5py.File(current_file) as hdf_file:
            eigenvalues.append(hdf_file[f"iteration{iteration}/eigenvalues"][:])
            condition_numbers.append(hdf_file[f"iteration{iteration}/condition_number"][()])

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,6))
        created_fig = True

    spacing = 0.5

    ax.boxplot(
        x=eigenvalues,
        positions= [i * spacing for i in range(len(eigenvalues))],
        tick_labels=range(len(eigenvalues)) if labels is None else labels,
        medianprops=dict(visible=False),
        patch_artist=True,
        boxprops=dict(facecolor='lightblue')
    )     
    for i, cond in enumerate(condition_numbers):
        ax.hlines(
            y=cond,        
            xmin=i * spacing - 0.1,     
            xmax=i * spacing  + 0.1,        
            colors="lightcoral",       
            linewidth=2,
            label="Condition number" if i == 0 else ""
        )
    # Labels and title
    ax.set_title("Eigenvalues")
    ax.set_xlabel("")
    ax.set_ylabel("Eigenvalues")

    ax.grid(True, which="major", axis="y", linestyle="--", linewidth=0.5)
    ax.grid(True, which="minor", axis="y", linestyle=":", linewidth=0.3)
    ax.grid(True, which="major", axis="x", linestyle="--", linewidth=0.5)

    if created_fig:
        ax.legend()
        plt.tight_layout()
        if output_file is not None:
            plt.savefig(output_file, dpi=500)
        plt.show()
    
    return ax

def evaluation_summary(
        file : list[str] | list[pl.Path] | str | pl.Path, 
    ):
    files = file
    if isinstance(file, (str, pl.Path)):
        files = [file]

    for current_file in files:
        with h5py.File(current_file) as hdf_file:
            pass    
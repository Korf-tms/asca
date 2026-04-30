# Additive Approximation of the Schur Complement

The implementation of ASCA (additive Schur complement approximation) that is based on the algorithm described in the work.

## Repository Structure
- `asca/` - implementation
- `main.py` - script that generates all of the test data and figures
- `matrices/` - the tested graphs in adjacency matrix form
  - `matrices/110x110.hdf5` - corresponds to the first test Test 1: Grid
  - `matrices/skirt.mtx` - corresponds to the second test Test 2: Skirt mesh
  - `matrices/annulus.mtx` - corresponds to the third test Test 3: Annulus mesh
  - `matrices/olafu.mtx` - corresponds to the fourth test Test 4: Structure mesh
  - `matrices/ct2010.mtx` - corresponds to the fifth test Test 5: U.S. Census 2010
  - `matrices/example.mtx` - this is the example graph used in visualizating the subgraph selection emthods Figures 4.3 and 4.4
- `data/` - generated ASCA approximation outputs
- `evaluation/` - generated evaluation results
- `schur_cache/` - cached exact Schur complements
- `figures/` - generated plots and figures
- `log.log` - log file
Generated folders are ignored by Git.

## Requirements

The project uses Python and the following main packages:

- `numpy`
- `scipy`
- `h5py`
- `joblib`
- `matplotlib`

Install them with:

```bash
pip install -r requirements.txt
```

## Usage

Example configurations are in `main.py`.

To run the full experiment script:

```bash
python main.py
```

The script runs ASCA on selected matrices, evaluates the results, and generates figures.

## ASCA Configuration

ASCA is configured with `AscaConfig`:

```python
from asca import AscaConfig, run_approximation

config = AscaConfig(
    filename="matrices/example.mtx",
    coarse_selection_method="mis",
    coarse_selection_method_arguments={"size": 1},
    subgraph_creation_method="depth",
    subgraph_creation_method_arguments={"size": 4},
    output_file="data/example.hdf5",
    iterations=1,
)

run_approximation(config)
```

Multiple method configurations can be run by passing lists of method names and argument dictionaries.

## Coarse Vertex Selection Methods

- `mis` - maximal independent set
- `mis_degree_asc` - MIS with vertices sorted by degree ascending
- `mis_degree_desc` - MIS with vertices sorted by degree descending
- `mis_strength_asc` - MIS with vertices sorted by strength descending
- `mis_strength_desc` - MIS with vertices sorted by strength ascending
- `moore` - independent set based on Moore neighborhoods

Common parameter:

- `size` - neighborhood size, default `1`

## Subgraph Creation Methods

- `depth` - creates depth-limited neighborhoods around coarse vertices
- `moore_all` - creates Moore neighborhoods around all vertices
- `moore_coarse` - creates Moore neighborhoods around coarse vertices
- `macrostructure` - groups micro-neighborhoods of coarse vertices into larger macrostructure subgraphs

Common parameter:

- `size` - neighborhood size

`macrostructure` parameters:

- `micro_size`
- `connection_depth`
- `merge_distance`

## Evaluation

Use `EvaluatorConfig` and `run_evaluation` to evaluate generated approximation files:

```python
from asca import EvaluatorConfig, run_evaluation

config = EvaluatorConfig(
    input_files=["data/example_mis_size1_depth_size4.hdf5"],
)

run_evaluation(config)
```

Evaluation computes CG convergence histories and generalized eigenvalue information.

## Visualization

Plotting helpers are available in `asca.visualization`:

- `cg_error_history`
- `cg_residual_history`
- `eigenvalues`
- `approximation`

These functions read generated HDF5 files and create matplotlib figures.

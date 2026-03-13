
# Additive Approximation of the Schur Complement

## Usage

Example usage in main.py

### Select coarse methods
- mis - coarse vertices are maximal independent set, parameters
	- size - int, default = 1
- moore - similar to mis, except the space between vertices isnt neighbourhood but moore neighbourhood, parameters
	- size - int, default = 1

### Create sugbraphs methods
- depth - creates subgraphs around coarse vertice with selected depth, parameters
	- max_depth - int, default = 2
- moore_all - creates subgraphs around all vertices with the shape of moore neighbourhood, parameters
	- size - int, default = 1
- moore_coarse - creates subgraphs around coarse vertices with the shape of moore neighbourhood, parameters
	- size - int, default = 1

### Test graphs

Can be generated with generate_grid_graphs in utils.py

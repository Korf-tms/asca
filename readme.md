# Additive Approximation of the Schur Complement
## Usage
Number of iterations and coarse graph selection strategy need to be changed in code in asca.py, currently set to 2 iterations and selecting coarse vertices by independent set with moore neighbourhood
```bash
python asca.py <filename>
```
### Profiler
sorted by tottime and saved in log.txt
```bash
python -m cProfile -s tottime asca.py <filename> > log.txt
```
### Test graphs
Can be generated with generate_grid_graphs in utils.py
### Other
* program has limited error handling, will crash if graph is too small for more iterations
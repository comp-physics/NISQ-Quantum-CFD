# Hybrid-QuantumCFD

This project aims to build a hybrid quantum-classical CFD solver for incompressible flow problems.


## CFD setup
- SIMPLE algorithm
- PISO algorithm


## Linear system
### Classical Solver
- ?

### Quantum Solver
- VQE
- VQLS
- QAOA
- HHL


## Benchmarks
- 2D Lid-driven cavity flow (Done) 

<p align="center">
    <img width=80% src=./gallery/cavity_flow.png>
</p>



- 2D Kármán vortex street (Unstarted)
- 2D Taylor-Green Vortex (Unstarted)


## Performance
### Runtime Profile
1. Run `python -m cProfile -o out.prof ./test.py ` and then  visualize it with `snakeviz out.prof `
2. Run `pycallgraph graphviz -- ./test.py` in the `profile` folder to generate a call graph 
3. Add  `@profile` before `def main():` then run the line by line profiler with `kernprof -l main.py`. One can visualize it with `python -m line_profiler -rmt "test.py.lprof"` 


### Memory Profile
1. Add  `@profile` before `def main():` then run `python -m memory_profiler test.py`
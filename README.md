# Hybrid-QuantumCFD

This project aims to build a hybrid quantum-classical CFD solver for incompressible flow problems.


## CFD setup
- SIMPLE 
- PISO 


## Linear system
### Classical Solver
- Jacobi
- Gauss-Seidel 
- Successive Over-relaxation (SOR)

### Quantum Solver
- VQE
- VQLS
- QAOA
- HHL


## Benchmarks
- [x] 2D Lid-driven cavity flow (Done) 

<div align="center">
<img src="https://github.com/comp-physics/Hybrid-QuantumCFD/blob/master/Gallery/cavity_flow.png" height="260px"> <img src="https://github.com/comp-physics/Hybrid-QuantumCFD/blob/master/Benchmark/Re100/UBench_Re100_Grid60x60.png" height="260px">
</div>

- [ ]  2D Kármán vortex street (Unstarted)

- [ ]  2D Taylor-Green Vortex (Unstarted)


## Performance
### Runtime Profile
1. Run `python -m cProfile -o out.prof ./test.py ` and then  visualize it with `snakeviz out.prof `
2. Run `pycallgraph graphviz -- ./test.py` in the `profile` folder to generate a call graph 
3. Add  `@profile` before `def main():` then run the line by line profiler with `kernprof -l main.py`. One can visualize it with `python -m line_profiler -rmt "test.py.lprof"` 


### Memory Profile
1. Add  `@profile` before `def main():` then run `python -m memory_profiler test.py`
## NISQ Quantum CFD

This project builds a hybrid quantum-classical CFD solver for incompressible flow problems in the NISQ regime.

## Linear system

### Classical Solver
- Jacobi
- Gauss-Seidel 
- Successive Over-relaxation (SOR)
- Penta-diagonal solver

### Quantum Solver
- VQE
- VQLS
- HHL

## 2D Lid-driven Cavity Benchmark

<div align="center">
<img src="https://github.com/comp-physics/NISQ-Quantum-CFD/blob/master/Benchmark/benchmark-Re100.pdf" height="300px"> 
</div>


## Performance

### Runtime Profile
1. Run `python -m cProfile -o out.prof ./test.py ` and then  visualize it with `snakeviz out.prof `
2. Run `pycallgraph graphviz -- ./test.py` in the `profile` folder to generate a call graph 
3. Add  `@profile` before `def main():` then run the line by line profiler with `kernprof -l main.py`. One can visualize it with `python -m line_profiler -rmt "test.py.lprof"` 


### Memory Profile

Add  `@profile` before `def main():` then run `python -m memory_profiler test.py`

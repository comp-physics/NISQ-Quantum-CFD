# Hybrid-QuantumCFD

This project builds a hybrid quantum-classical CFD solver for incompressible flow problems in the NISQ regime.

## Linear system

### Classical Solver
- [x] Jacobi
- [x] Gauss-Seidel 
- [x] Successive Over-relaxation (SOR)
- [x] Penta-diagonal solver

### Quantum Solver
- [x] VQE
- [x] VQLS
- [x] HHL

## 2D Lid-driven Cavity Benchmark

<div align="center">
<img src="https://github.com/comp-physics/Hybrid-QuantumCFD/blob/master/Benchmark/Re100/Stream_Re100_Grid20x20.png" height="300px"> <img src="https://github.com/comp-physics/Hybrid-QuantumCFD/blob/master/Benchmark/Re100/UBench_Re100_Grid60x60.png" height="300px">
</div>

## Performance

### Runtime Profile
1. Run `python -m cProfile -o out.prof ./test.py ` and then  visualize it with `snakeviz out.prof `
2. Run `pycallgraph graphviz -- ./test.py` in the `profile` folder to generate a call graph 
3. Add  `@profile` before `def main():` then run the line by line profiler with `kernprof -l main.py`. One can visualize it with `python -m line_profiler -rmt "test.py.lprof"` 


### Memory Profile

Add  `@profile` before `def main():` then run `python -m memory_profiler test.py`

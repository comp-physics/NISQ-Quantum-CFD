## Incompressible Navier-Stokes solve on noisy quantum hardware via a hybrid quantum-classical scheme

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)

[Preprint here (arXiv)](https://arxiv.org/abs/2406.00280)

__Authors:__ Zhixin Song, Robert Deaton, Bryan Gard, Spencer H. Bryngelson

> Partial differential equation solvers are required to solve the Navier-Stokes equations for fluid flow. Recently, algorithms have been proposed to simulate fluid dynamics on quantum computers. Fault-tolerant quantum devices might enable exponential speedups over algorithms on classical computers. However, current and upcoming quantum hardware presents noise in the computations, requiring algorithms that make modest use of quantum resources: shallower circuit depths and fewer qubits. Variational algorithms are more appropriate and robust under resource restrictions. This work presents a hybrid quantum-classical algorithm for the incompressible Navier-Stokes equations. Classical devices perform nonlinear computations, and quantum ones use variational algorithms to solve the pressure Poisson equation. A lid-driven cavity problem benchmarks the method. We verify the algorithm via noise-free simulation and test it on noisy IBM superconducting quantum hardware. Results show that high-fidelity results can be achieved via this approach, even on current quantum devices. A multigrid preconditioning approach helps avoid local minima. HTree, a tomography technique with linear complexity in qubit count, reduces the quantum state readout time. We compare the quantum resources required for near-term and fault-tolerant solvers to determine quantum hardware requirements for fluid simulations with complexity improvements.


### Linear system

#### Classical Solver
- Jacobi
- Gauss-Seidel 
- Successive Over-relaxation (SOR)
- Penta-diagonal solver

#### Quantum Solver
- VQE
- VQLS
- HHL

### 2D Lid-driven Cavity Benchmark

<div align="center">
<img src="https://github.com/comp-physics/NISQ-Quantum-CFD/blob/master/Benchmark/benchmark-Re100.png" height="300px"> 
</div>


### Performance

#### Runtime Profile

1. Run `python -m cProfile -o out.prof ./test.py ` and then  visualize it with `snakeviz out.prof `
2. Run `pycallgraph graphviz -- ./test.py` in the `profile` folder to generate a call graph 
3. Add  `@profile` before `def main():` then run the line by line profiler with `kernprof -l main.py`. One can visualize it with `python -m line_profiler -rmt "test.py.lprof"` 


### Memory Profile

Add  `@profile` before `def main():` then run `python -m memory_profiler test.py`

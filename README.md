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

![fig1](./gallery/cavity_flow.png)

- 2D Kármán vortex street (Unstarted)
- 2D Taylor-Green Vortex (Unstarted)


## Performance
### Runtime Profile
1. Run `python -m cProfile -o out.prof ./test.py ` and then  visualize it with `snakeviz out.prof `
2. Run `pycallgraph graphviz -- ./test.py` in the `profile` folder to generate a call graph 


### Memory Profile
1. 
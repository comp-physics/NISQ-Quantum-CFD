import numpy as np
from scipy import sparse
from utils import u_momentum, v_momentum, get_rhs, get_coeff_mat, pressure_correct, update_velocity,  check_divergence_free
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import Isometry
from qiskit.quantum_info import Pauli
from qiskit.quantum_info.operators import Operator, SparsePauliOp
from pyamg.aggregation import smoothed_aggregation_solver


def generate_projection_ham(space_size, N, time_size, iteration):
    
    Lx = 1           # length of domian in x-direction
    Ly = 1           # length of domian in y-direction
    nx = space_size  # grid size in x-direction
    ny = space_size  # grid size in y-direction

    Re = 100         # Reynolds number
    nu = 1 / Re      # kinematic viscosity
    rho = 1.0        # density
    dt = time_size   # time step size
    itr = iteration  # iterations
    velocity = 1.0


    # Create staggered grid index system
    imin, jmin = 1, 1
    imax = imin + nx - 1
    jmax = jmin + ny - 1

    # Define ghost cells for boundary conditions
    x = np.zeros(jmax + 2)
    y = np.zeros(jmax + 2)
    x[imin: imax + 2] = np.linspace(0, Lx, nx + 1, endpoint=True)
    y[jmin: jmax + 2] = np.linspace(0, Ly, ny + 1, endpoint=True)

    dx = x[imin + 1] - x[imin]
    dy = y[jmin + 1] - y[jmin]
    dxi = 1 / dx
    dyi = 1 / dy


    # Define Laplacian
    def Laplacian(nx, ny, dxi, dyi):
        Dx = np.diag(np.ones(nx)) * 2 - np.diag(np.ones(nx - 1), 1) - np.diag(np.ones(nx - 1), -1)
        Dx[0, 0] = 1
        Dx[-1, -1] = 1
        Ix = np.diag(np.ones(ny))
        Dy = np.diag(np.ones(ny)) * 2 - np.diag(np.ones(ny - 1), 1) - np.diag(np.ones(ny - 1), -1)
        Dy[0, 0] = 1
        Dy[-1, -1] = 1
        Iy = np.diag(np.ones(nx))
        L = np.kron(Ix, Dx) * dxi ** 2 + np.kron(Dy, Iy) * dyi ** 2
        return L

    L = Laplacian(nx, ny, dxi, dyi)
    L[0, :] = 0
    L[0, 0] = 1

    X = SparsePauliOp(Pauli('X'))
    Y = SparsePauliOp(Pauli('Y'))
    Z = SparsePauliOp(Pauli('Z'))
    
    def Ident(dim):
        if dim ==0:
            return 1
        else:
            term = Pauli('I')
            for i in range(dim-1):
                term = term^Pauli('I')
            return SparsePauliOp(term)
    
    def complement_eye(qubit):
        ident1 = np.eye(imax*jmax//2)
        proj = np.array([[1,0],[0,0]])
        if qubit == 0:
            res = np.kron(proj,ident1)
        else:
            ident1 = np.eye(2**qubit)
            ident2 = np.eye(imax*jmax//(2*int(np.sqrt(np.size(ident1)))))
            res = np.kron(np.kron(ident1,proj),ident2)
        return res

    # Variable declaration
    u = np.zeros((imax + 2, jmax + 2))
    v = np.zeros((imax + 2, jmax + 2))

    for _ in range(itr):
        # u-momentum
        us = u.copy()
        u_old = u.copy()
        I = slice(imin + 1, imax + 1)
        Ib = slice(imin, imax)
        If = slice(imin + 2, imax + 2)

        J = slice(jmin, jmax + 1)
        Jb = slice(jmin - 1, jmax)
        Jf = slice(jmin + 1, jmax + 2)

        v_here = 0.25 * (v[Ib, J] + v[Ib, Jf] + v[I, J] + v[I, Jf])
        us[I, J] = u[I, J] + dt * (
                nu * (u[Ib, J] - 2 * u[I, J] + u[If, J]) * dxi ** 2
                + nu * (u[I, Jb] - 2 * u[I, J] + u[I, Jf]) * dyi ** 2
                - v_here * (u[I, Jf] - u[I, Jb]) * 0.5 * dyi
                - u[I, J] * (u[If, J] - u[Ib, J]) * 0.5 * dxi
        )

        # v-momentum
        vs = v.copy()
        v_old = v.copy()
        I = slice(imin, imax + 1)
        Ib = slice(imin - 1, imax)
        If = slice(imin + 1, imax + 2)

        J = slice(jmin + 1, jmax + 1)
        Jb = slice(jmin, jmax)
        Jf = slice(jmin + 2, jmax + 2)

        u_here = 0.25 * (u[I, Jb] + u[I, J] + u[If, Jb] + u[If, J])
        vs[I, J] = v[I, J] + dt * (
                nu * (v[Ib, J] - 2 * v[I, J] + v[If, J]) * dxi ** 2
                + nu * (v[I, Jb] - 2 * v[I, J] + v[I, Jf]) * dyi ** 2
                - u_here * (v[If, J] - v[Ib, J]) * 0.5 * dxi
                - v[I, J] * (v[I, Jf] - v[I, Jb]) * 0.5 * dyi
        )

        # Claculate R.H.S of pressure Poisson
        Rn = -rho / dt * ((us[imin + 1: imax + 2, jmin: jmax + 1]
                           - us[imin: imax + 1, jmin: jmax + 1]) * dxi
                          + (vs[imin: imax + 1, jmin + 1: jmax + 2]
                             - vs[imin: imax + 1, jmin: jmax + 1]) * dyi)

        R = Rn.T.ravel()

        # Prepare linear system for VQE
        pv = np.linalg.solve(L, R)

        pn = pv.reshape(ny, nx).T
        p = np.zeros((imax + 1, jmax + 1))
        p[1:, 1:] = pn
        p[0, 0] = 0

        # Correct velocity
        u[imin + 1: imax + 1, jmin: jmax + 1] = us[imin + 1: imax + 1, jmin: jmax + 1] - dt / rho * (
                p[imin + 1: imax + 1, jmin: jmax + 1] - p[imin: imax, jmin: jmax + 1]) * dxi
        v[imin: imax + 1, jmin + 1: jmax + 1] = vs[imin: imax + 1, jmin + 1: jmax + 1] - dt / rho * (
                p[imin: imax + 1, jmin + 1: jmax + 1] - p[imin: imax + 1, jmin: jmax]) * dyi

        # Update BCs
        v[imin, :] = 0.0  # left wall
        v[imax + 1, :] = 0.0  # right wall
        v[:, jmin - 1] = -v[:, jmin]  # bottom wall
        v[:, jmax + 1] = -v[:, jmax]  # top wall

        u[imin - 1, :] = -u[imin, :]  # left wall
        u[imax + 1, :] = -u[imax, :]  # right wall
        u[:, imin] = 0  # bottom wall
        u[:, imax + 1] = velocity  # top wall

    ##==================== Build Hamiltonian ======================================
    b = R / np.linalg.norm(R)
    x = np.linalg.solve(L, b)
    x_norm = np.linalg.norm(x)
    x_normalized = x / np.linalg.norm(x)

    dim = 2**N  # dimension of the operator A
    b = b.reshape([dim, 1])
    H_test = L.conj().T @ (np.eye(dim) - b @ b.T) @ L
    H_test = H_test/np.linalg.norm(H_test)

    # ml = smoothed_aggregation_solver(L) 
    # op = ml.aspreconditioner(cycle='V')
    # M = op * np.identity(op.shape[1])
    # L_new = M@L
    # b_new = M@b
    # b_new = b_new/np.linalg.norm(b_new)
    # H_amg = L_new.conj().T@(np.eye(dim)- b_new@b_new.T)@L_new

    M = np.linalg.inv(np.diag(np.diag(L)))
    L_new = M@L
    b_new = M@b
    b_new = b_new /np.linalg.norm(b_new)
    H_jacobi = L_new.conj().T@(np.eye(dim)- b_new@b_new.T)@L_new

    # Local cost function
    qc = QuantumCircuit(N)
    qc.append(Isometry(b_new, num_ancillas_zero=0, num_ancillas_dirty=0), qargs=qc.qregs[0])
    qc.save_unitary()
    
    # Transpile for simulator
    simulator = AerSimulator(method = 'unitary')
    circ = transpile(qc, simulator)
    
    # Run and get unitary
    result = simulator.run(circ).result()
    unitary = result.get_unitary(circ)
    U = np.asarray(unitary)
    sum_piece =  np.zeros((2**N,2**N))
    
    for q in range(N):
        sum_piece += complement_eye(q)
    H_local = L_new.conj().T @ U @ ((Ident(N).to_matrix() - 1/(N) * sum_piece)) @ U.conj().T @ L_new

    return H_test, H_jacobi, H_local, x_normalized


def generate_simple_ham(space_size, N, iteration):
    
    imax = space_size          # grid size in x-direction
    jmax = space_size          # grid size in y-direction
    max_iteration = iteration  # iterations
    max_res = 1000             # initial residual
    rho = 1                    # density
    velocity = 1               # velocity = lid velocity
    Re = 100                   # Reynolds number
    mu = 1 / Re                # kinematic viscosity = 1/Reynolds number
    Lx = 1
    Ly = 1
    dx = Lx / (imax - 1)       # dx, dy cell sizes along x and y directions
    dy = Ly / (jmax - 1)

    x = np.arange(dx / 2, Lx, dx)
    y = np.arange(0, Ly + dy / 2, dy)
    alpha_p = 0.1              # pressure under-relaxation
    alpha_u = 0.7              # velocity under-relaxation
    tol = 1e-5                 # tolerance for convergence

    # Variable declaration
    p = np.zeros((imax, jmax))         # p = Pressure
    p_star = np.zeros((imax, jmax))    # intermediate pressure
    p_prime = np.zeros((imax, jmax))   # pressure correction
    rhsp = np.zeros((imax, jmax))      # right hand side vector of pressure correction equation
    div = np.zeros((imax, jmax))

    # Vertical velocity
    v_star = np.zeros((imax, jmax + 1))  # intermediate velocity
    v = np.zeros((imax, jmax + 1))       # final velocity
    d_v = np.zeros((imax, jmax + 1))     # velocity correction coefficient

    # Horizontal Velocity
    u_star = np.zeros((imax + 1, jmax))  # intermediate velocity
    u = np.zeros((imax + 1, jmax))       # final velocity
    d_u = np.zeros((imax + 1, jmax))     # velocity correction coefficient

    # Boundary condition
    # Lid velocity (Top wall is moving with 1m/s)
    u_star[:, jmax - 1] = velocity
    u[:, jmax - 1] = velocity

    iteration = 1
    # max_iteration = 10    # for debug
    while iteration <= max_iteration and max_res > tol:
        iteration += 1

        # Solve u and v momentum equations for intermediate velocity
        u_star, d_u = u_momentum(imax, jmax, dx, dy, rho, mu, u, v, p_star, velocity, alpha_u)
        v_star, d_v = v_momentum(imax, jmax, dx, dy, rho, mu, u, v, p_star, alpha_u)

        u_old = u.copy()
        v_old = v.copy()

        # Calculate rhs vector of the Pressure Poisson matrix
        rhsp = get_rhs(imax, jmax, dx, dy, rho, u_star, v_star)

        # Form the Pressure Poisson coefficient matrix
        Ap = get_coeff_mat(imax, jmax, dx, dy, rho, d_u, d_v)

        # Solve pressure correction implicitly and update pressure
        p, p_prime, p_prime_interior = pressure_correct(imax, jmax, rhsp, Ap, p_star, alpha_p)

        # Update velocity based on pressure correction
        u, v = update_velocity(imax, jmax, u_star, v_star, p_prime, d_u, d_v, velocity)

        # Check if velocity field is divergence free
        div = check_divergence_free(imax, jmax, dx, dy, u, v)

        p_star = p.copy()  # use p as p_star for the next iteration

    ##==================== Build Hamiltonian ======================================
    R = rhsp
    L = Ap

    b = R / np.linalg.norm(R)
    x = np.linalg.solve(L, b)
    x_norm = np.linalg.norm(x)
    x_normalized = x / np.linalg.norm(x)

    dim = 2**N  # dimension of the operator A
    b = b.reshape([dim, 1])
    H_test = L.conj().T @ (np.eye(dim) - b @ b.T) @ L

    # ml = smoothed_aggregation_solver(L) 
    # op = ml.aspreconditioner(cycle='V')
    # M = op * np.identity(op.shape[1])
    # L_new = M@L
    # b_new = M@b
    # b_new = b_new/np.linalg.norm(b_new)
    # H_amg = L_new.conj().T@(np.eye(dim)- b_new@b_new.T)@L_new

    M = np.linalg.inv(np.diag(np.diag(L)))
    L_new = M@L
    b_new = M@b
    b_new = b_new /np.linalg.norm(b_new)
    H_jacobi = L_new.conj().T@(np.eye(dim)- b_new@b_new.T)@L_new

    return H_test, H_jacobi, x_normalized
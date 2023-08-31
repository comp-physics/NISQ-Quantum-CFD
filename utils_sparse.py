import numpy as np
from scipy import sparse
from scipy.sparse import lil_matrix as zeros
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix

__all__ = [
    "u_momentum",
    "v_momentum",
    "get_rhs",
    "get_coeff_mat_modified",
    "pres_correct",
    "update_velocity",
    "check_divergence_free"
]

def u_momentum(imax, jmax, dx, dy, rho, mu, u, v, p, velocity, alpha):
    u_star = zeros((imax + 1, jmax))
    d_u = zeros((imax + 1, jmax))

    De = mu * dy / dx  # convective coefficients
    Dw = mu * dy / dx
    Dn = mu * dx / dy
    Ds = mu * dx / dy

    def A(F, D):
        return max(0, (1 - 0.1 * abs(F / D))**5)

    # compute u_star
    for i in range(1, imax):
        for j in range(1, jmax - 1):
            Fe = 0.5 * rho * dy * (u[i + 1, j] + u[i, j])
            Fw = 0.5 * rho * dy * (u[i - 1, j] + u[i, j])
            Fn = 0.5 * rho * dx * (v[i, j + 1] + v[i - 1, j + 1])
            Fs = 0.5 * rho * dx * (v[i, j] + v[i - 1, j])

            aE = De * A(Fe, De) + max(-Fe, 0)
            aW = Dw * A(Fw, Dw) + max(Fw, 0)
            aN = Dn * A(Fn, Dn) + max(-Fn, 0)
            aS = Ds * A(Fs, Ds) + max(Fs, 0)
            aP = aE + aW + aN + aS + (Fe - Fw) + (Fn - Fs)

            pressure_term = (p[i - 1, j] - p[i, j]) * dy

            u_star[i, j] = alpha / aP * ((aE * u[i + 1, j] + aW * u[i - 1, j] + aN * u[i, j + 1] + aS * u[i, j - 1]) + pressure_term) + (1 - alpha) * u[i, j]

            d_u[i, j] = alpha * dy / aP  # refer to Versteeg CFD book

    # set d_u for top and bottom BCs
    for i in range(1, imax):
        j = 0  # bottom
        Fe = 0.5 * rho * dy * (u[i + 1, j] + u[i, j])
        Fw = 0.5 * rho * dy * (u[i - 1, j] + u[i, j])
        Fn = 0.5 * rho * dx * (v[i, j + 1] + v[i - 1, j + 1])
        Fs = 0

        aE = De * A(Fe, De) + max(-Fe, 0)
        aW = Dw * A(Fw, Dw) + max(Fw, 0)
        aN = Dn * A(Fn, Dn) + max(-Fn, 0)
        aS = 0
        aP = aE + aW + aN + aS + (Fe - Fw) + (Fn - Fs)
        d_u[i, j] = alpha * dy / aP

        j = jmax - 1  # top
        Fe = 0.5 * rho * dy * (u[i + 1, j] + u[i, j])
        Fw = 0.5 * rho * dy * (u[i - 1, j] + u[i, j])
        Fn = 0
        Fs = 0.5 * rho * dx * (v[i, j] + v[i - 1, j])

        aE = De * A(Fe, De) + max(-Fe, 0)
        aW = Dw * A(Fw, Dw) + max(Fw, 0)
        aN = 0
        aS = Ds * A(Fs, Ds) + max(Fs, 0)
        aP = aE + aW + aN + aS + (Fe - Fw) + (Fn - Fs)
        d_u[i, j] = alpha * dy / aP

    # Apply BCs
    u_star[0, :jmax] = -u_star[1, :jmax]  # left wall
    u_star[imax, :jmax] = -u_star[imax - 1, :jmax]  # right wall
    u_star[:, 0] = 0.0  # bottom wall
    u_star[:, jmax - 1] = velocity  # top wall

    return u_star, d_u

def v_momentum(imax, jmax, dx, dy, rho, mu, u, v, p, alpha):

    v_star = zeros((imax, jmax+1))
    d_v = zeros((imax, jmax+1))

    De = mu * dy / dx  # convective coefficients
    Dw = mu * dy / dx
    Dn = mu * dx / dy
    Ds = mu * dx / dy

    A = lambda F, D: max(0, (1-0.1 * abs(F/D))**5)

    # compute u_star
    for i in range(1, imax-1):
        for j in range(1, jmax):
            Fe = 0.5 * rho * dy * (u[i+1, j] + u[i+1, j-1])
            Fw = 0.5 * rho * dy * (u[i, j] + u[i, j-1])
            Fn = 0.5 * rho * dx * (v[i, j] + v[i, j+1])
            Fs = 0.5 * rho * dx * (v[i, j-1] + v[i, j])

            aE = De * A(Fe, De) + max(-Fe, 0)
            aW = Dw * A(Fw, Dw) + max(Fw, 0)
            aN = Dn * A(Fn, Dn) + max(-Fn, 0)
            aS = Ds * A(Fs, Ds) + max(Fs, 0)
            aP = aE + aW + aN + aS + (Fe - Fw) + (Fn - Fs)

            pressure_term = (p[i, j-1] - p[i, j]) * dx

            v_star[i, j] = alpha / aP * (aE * v[i+1, j] + aW * v[i-1, j] + aN * v[i, j+1] + aS * v[i, j-1] + pressure_term) + (1-alpha) * v[i, j]

            d_v[i, j] = alpha * dx / aP  # refer to Versteeg CFD book

    # set d_v for left and right BCs
    # Apply BCs
    for j in range(1, jmax):
        i = 0  # left BC
        Fe = 0.5 * rho * dy * (u[i+1, j] + u[i+1, j-1])
        Fw = 0
        Fn = 0.5 * rho * dx * (v[i, j] + v[i, j+1])
        Fs = 0.5 * rho * dx * (v[i, j-1] + v[i, j])

        aE = De * A(Fe, De) + max(-Fe, 0)
        aW = 0
        aN = Dn * A(Fn, Dn) + max(-Fn, 0)
        aS = Ds * A(Fs, Ds) + max(Fs, 0)
        aP = aE + aW + aN + aS + (Fe - Fw) + (Fn - Fs)
        d_v[i, j] = alpha * dx / aP

        i = imax - 1  # right BC
        Fe = 0
        Fw = 0.5 * rho * dy * (u[i, j] + u[i, j-1])
        Fn = 0.5 * rho * dx * (v[i, j] + v[i, j+1])
        Fs = 0.5 * rho * dx * (v[i, j-1] + v[i, j])

        aE = 0
        aW = Dw * A(Fw, Dw) + max(Fw, 0)
        aN = Dn * A(Fn, Dn) + max(-Fn, 0)
        aS = Ds * A(Fs, Ds) + max(Fs, 0)
        aP = aE + aW + aN + aS + (Fe - Fw) + (Fn - Fs)
        d_v[i, j] = alpha * dx / aP

    # apply BCs
    v_star[0, :] = 0.0  # left wall
    v_star[imax-1, :] = 0.0  # right wall
    v_star[:, 0] = -v_star[:, 1]  # bottom wall
    v_star[:, jmax] = -v_star[:, jmax-1]  # top wall

    return v_star, d_v


def get_rhs(imax, jmax, dx, dy, rho, u_star, v_star):

    stride = jmax
    bp = zeros((jmax* imax,1))
    # print(bp)
    # RHS is the same for all nodes except the first one
    # because the first element is set to be zero, it has no pressure correction
    for j in range(jmax):
        for i in range(imax):
            position = i + j * stride
            #print(position)
            bp[position,0] = rho * (u_star[i,j] * dy - u_star[i+1,j] * dy + v_star[i,j] * dx - v_star[i,j+1] * dx)
    # modify for the first element
    bp[0,0] = 0

    return bp

def get_coeff_mat_modified(imax, jmax, dx, dy, rho, d_u, d_v):

    N = imax * jmax
    stride = jmax
    Ap = csr_matrix((N, N))

    for j in range(jmax):
        for i in range(imax):
            position = i + j * stride
            aE, aW, aN, aS = 0, 0, 0, 0

            # Set BCs for four corners
            if i == 0 and j == 0:
                Ap[position, position] = 1
                continue

            if i == imax-1 and j == 0:
                Ap[position, position-1] = -rho * d_u[i,j] * dy
                aW = -Ap[position, position-1]
                Ap[position, position+stride] = -rho * d_v[i,j+1] * dx
                aN = -Ap[position, position+stride]
                Ap[position, position] = aE + aN + aW + aS
                continue

            if i == 0 and j == jmax-1:
                Ap[position, position+1] = -rho * d_u[i+1,j] * dy
                aE = -Ap[position, position+1]
                Ap[position, position-stride] = -rho * d_v[i,j] * dx
                aS = -Ap[position, position-stride]
                Ap[position, position] = aE + aN + aW + aS
                continue

            if i == imax-1 and j == jmax-1:
                Ap[position, position-1] = -rho * d_u[i,j] * dy
                aW = -Ap[position, position-1]
                Ap[position, position-stride] = -rho * d_v[i,j] * dx
                aS = -Ap[position, position-stride]
                Ap[position, position] = aE + aN + aW + aS
                continue

            # Set four boundaries
            if i == 0:
                Ap[position, position+1] = -rho * d_u[i+1,j] * dy
                aE = -Ap[position, position+1]
                Ap[position, position+stride] = -rho * d_v[i,j+1] * dx
                aN = -Ap[position, position+stride]
                Ap[position, position-stride] = -rho * d_v[i,j] * dx
                aS = -Ap[position, position-stride]
                Ap[position, position] = aE + aN + aW + aS
                continue

            if j == 0:
                Ap[position, position+1] = -rho * d_u[i+1,j] * dy
                aE = -Ap[position, position+1]
                Ap[position, position+stride] = -rho * d_v[i,j+1] * dx
                aN = -Ap[position, position+stride]
                Ap[position, position-1] = -rho * d_u[i,j] * dy
                aW = -Ap[position, position-1]
                Ap[position, position] = aE + aN + aW + aS
                continue

            if i == imax-1:
                Ap[position, position+stride] = -rho * d_v[i,j+1] * dx
                aN = -Ap[position, position+stride]
                Ap[position, position-stride] = -rho * d_v[i,j] * dx
                aS = -Ap[position, position-stride]
                Ap[position, position-1] = -rho * d_u[i,j] * dy
                aW = -Ap[position, position-1]
                Ap[position, position] = aE + aN + aW + aS
                continue

            if j == jmax-1:
                Ap[position, position+1] = -rho * d_u[i+1,j] * dy
                aE = -Ap[position, position+1]
                Ap[position, position-stride] = -rho * d_v[i,j] * dx
                aS = -Ap[position, position-stride]
                Ap[position, position-1] = -rho * d_u[i,j] * dy
                aW = -Ap[position, position-1]
                Ap[position, position] = aE + aN + aW + aS
                continue

            # Interior nodes
            Ap[position, position-1] = -rho * d_u[i,j] * dy
            aW = -Ap[position, position-1]

            Ap[position, position+1] = -rho * d_u[i+1,j] * dy
            aE = -Ap[position, position+1]

            Ap[position, position-stride] = -rho * d_v[i,j] * dx
            aS = -Ap[position, position-stride]

            Ap[position, position+stride] = -rho * d_v[i,j+1] * dx
            aN = -Ap[position, position+stride]

            Ap[position, position] = aE + aN + aW + aS

    return Ap


def pres_correct(imax, jmax, rhsp, Ap, p, alpha):
    pressure = lil_matrix(p)  # Initial pressure
    p_prime = lil_matrix((imax, jmax))  # Pressure correction matrix
    p_prime_interior = sparse.linalg.spsolve(Ap, rhsp)


    z = 0  # Adjusted the indexing to start from 0
    for j in range(jmax):
        for i in range(imax):
            p_prime[i, j] = p_prime_interior[z]
            z += 1
            pressure[i, j] = p[i, j] + alpha * p_prime[i, j]

    pressure[0, 0] = 0  # Set the pressure at the first node to zero

    return pressure, p_prime


def update_velocity(imax, jmax, u_star, v_star, p_prime, d_u, d_v, velocity):
    u = zeros((imax+1, jmax))
    v = zeros((imax, jmax+1))

    # Update interior nodes of u and v
    for i in range(1, imax):
        for j in range(1, jmax-1):
            u[i,j] = u_star[i,j] + d_u[i,j] * (p_prime[i-1,j] - p_prime[i,j])

    for i in range(1, imax-1):
        for j in range(1, jmax):
            v[i,j] = v_star[i,j] + d_v[i,j] * (p_prime[i,j-1] - p_prime[i,j])

    # Update BCs
    v[0,:] = 0.0          # left wall
    v[imax-1,:] = 0.0     # right wall
    v[:,0] = -v[:,1]      # bottom wall
    v[:,-1] = -v[:,-2]    # top wall

    u[0,:] = -u[1,:]      # left wall
    u[imax,:] = -u[imax-1,:] # right wall
    u[:,0] = 0.0          # bottom wall
    u[:,-1] = velocity    # top wall

    return u, v


def check_divergence_free(imax, jmax, dx, dy, u, v):
    div = zeros((imax, jmax))

    for i in range(imax):
        for j in range(jmax):
            div[i, j] = (1/dx) * (u[i, j] - u[i+1, j]) + (1/dy) * (v[i, j] - v[i, j+1])

    return div




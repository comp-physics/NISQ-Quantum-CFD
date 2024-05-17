# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Hamiltonian simulation of 2d discrete Laplace operator."""

from typing import Tuple
import numpy as np
import scipy as sp

from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.quantum_info.operators import SparsePauliOp, Operator
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter, SuzukiTrotter

from .linear_system_matrix import LinearSystemMatrix


class DiscreteLaplacian(LinearSystemMatrix):

    # Fix this comment

    r"""Class of tridiagonal Toeplitz symmetric matrices.

    Given the main entry, :math:`a`, and the off diagonal entry, :math:`b`, the :math:`4\times 4`
    dimensional tridiagonal Toeplitz symmetric matrix is

    .. math::

        \begin{pmatrix}
            a & b & 0 & 0 \\
            b & a & b & 0 \\
            0 & b & a & b \\
            0 & 0 & b & a
        \end{pmatrix}.

    Examples:

        .. jupyter-execute::

            import numpy as np
            from qiskit import QuantumCircuit
            from qiskit.algorithms.linear_solvers.matrices import TridiagonalToeplitz

            matrix = TridiagonalToeplitz(2, 1, -1 / 3)
            power = 3

            # Controlled power (as within QPE)
            num_qubits = matrix.num_state_qubits
            pow_circ = matrix.power(power).control()
            circ_qubits = pow_circ.num_qubits
            qc = QuantumCircuit(circ_qubits)
            qc.append(matrix.power(power).control(), list(range(circ_qubits)))
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        boundary: bool = True,
        trotterized: bool = True,
        tolerance: float = 1e-2,
        evolution_time: float = 1.0,
        trotter_steps: int = 1,
        trotter_order: int = 1,
        name: str = "tridi",
    ) -> None:
        """
        Args:
            nx: the number of cells in the x-direction of the discretization
            ny: the number of cells in the x-direction of the discretization
            boundary: whether or not the Laplacian has boundary conditions (as in the lid-driven cavity flow problem)
            trotterized: whether or not to represent the hamiltonian evolution using Lie-Trotter or raw unitary
            tolerance: the accuracy desired for the approximation
            evolution_time: the time of the Hamiltonian simulation
            trotter_steps: the number of Trotter steps
            trotter_order: the order of the Lie-Suzuki-Trotter decomposition to represent the hamiltonian evolution
            name: The name of the object.
        """
        # define internal parameters
        self._num_state_qubits = None
        self._nx = None
        self._ny = None
        self._boundary = None
        self._trotterized = None
        self._tolerance = None
        self._evolution_time = None  # makes sure the eigenvalues are contained in [0,1)
        self._trotter_steps = None
        self._trotter_order = None

        # store parameters
        self.nx = nx
        self.ny = ny
        self.boundary = boundary
        self.trotterized = trotterized
        self.trotter_order = trotter_order
        super().__init__(
            num_state_qubits=self.nx + self.ny + self.boundary,
            tolerance=tolerance,
            evolution_time=evolution_time,
            name=name,
        )
        self.trotter_steps = trotter_steps

    @property
    def num_state_qubits(self) -> int:
        r"""The number of state qubits representing the state :math:`|x\rangle`.

        Returns:
            The number of state qubits.
        """
        return self._num_state_qubits

    @num_state_qubits.setter
    def num_state_qubits(self, num_state_qubits: int) -> None:
        """Set the number of state qubits.

        Note that this may change the underlying quantum register, if the number of state qubits
        changes.

        Args:
            num_state_qubits: The new number of qubits.
        """
        if num_state_qubits != self._num_state_qubits:
            self._invalidate()
            self._num_state_qubits = num_state_qubits
            self._reset_registers(num_state_qubits)

    @property
    def nx(self) -> int:
        """Return log2 of the grid size in the x-direction."""
        return self._nx

    @nx.setter
    def nx(self, nx: int) -> None:
        """Set the log2 of the grid width in the x-direction."""
        self._nx = nx

    @property
    def ny(self) -> int:
        """Return log2 of the grid size in the x-direction."""
        return self._yx

    @nx.setter
    def ny(self, ny: int) -> None:
        """Set the log2 of the grid width in the y-direction."""
        self._ny = ny

    @property
    def boundary(self) -> bool:
        """Return whether boundary conditions are set."""
        return self._boundary

    @boundary.setter
    def boundary(self, boundary: bool) -> None:
        """Set whether boundary conditions for the Lid-Driven Cavity flow problem are active.
            This fixes the first row of the Laplacian matrix to 1 followed by all 0's and makes L non-hermitian.
            To account for this, we use an extra qubit and write C = (0 L ; L* 0) as a block anti-diagonal hermitian
            matrix and solve C(0 ; x) = (b ; 0)."""

        self._boundary = boundary

    @property
    def trotterized(self) -> bool:
        """Return whether a SparsePauli decomposition and Lie-Trotter evolution is used to encode the
            Hamiltonian evolution. If False, a unitary gate is computed as a matrix exponential and encoded
            with qc.unitary, which is what HHL defaults to when a generic matrix is used."""
        return self._trotterized

    @trotterized.setter
    def trotterized(self, trotterized: bool) -> None:
        """Set whether Lie-Trotter evolution is used (if False, unitary gate)"""
        self._trotterized = trotterized

    @property
    def tolerance(self) -> float:
        """Return the error tolerance"""
        return self._tolerance

    @tolerance.setter
    def tolerance(self, tolerance: float) -> None:
        """Set the error tolerance.
        Args:
            tolerance: The new error tolerance.
        """
        self._tolerance = tolerance

    @property
    def evolution_time(self) -> float:
        """Return the time of the evolution."""
        return self._evolution_time

    @evolution_time.setter
    def evolution_time(self, evolution_time: float) -> None:
        """Set the time of the evolution.
        Args:
            evolution_time: The new time of the evolution.
        """
        self._evolution_time = evolution_time

    @property
    def trotter_steps(self) -> int:
        """Return the number of trotter steps."""
        return self._trotter_steps

    @trotter_steps.setter
    def trotter_steps(self, trotter_steps: int) -> None:
        """Set the number of trotter steps.
        Args:
            trotter_steps: The new number of trotter steps.
        """
        self._trotter_steps = trotter_steps

    @property
    def trotter_order(self) -> int:
        """Return the order of the lie-suzuki-trotter formula in the hamiltonian evolution."""
        return self._trotter_order

    @trotter_order.setter
    def trotter_order(self, trotter_order: int) -> None:
        """Set the order of the lie-suzuki-trotter formula in the hamiltonian evolution.
        Args:
            trotter_steps: The new order of the lie-suzuki-trotter formula.
        """
        self._trotter_order = trotter_order

    @property
    def matrix(self) -> np.ndarray:
        """Returns the discretized Laplacian."""
        nx = 2**self.nx
        ny = 2**self.ny
        Dx = np.diag(np.ones(nx)) * 2 - np.diag(np.ones(nx - 1), 1) - np.diag(np.ones(nx - 1), -1)
        Dx[0, 0] = 1
        Dx[-1, -1] = 1
        Ix = np.diag(np.ones(ny))
        Dy = np.diag(np.ones(ny)) * 2 - np.diag(np.ones(ny - 1), 1) - np.diag(np.ones(ny - 1), -1)
        Dy[0, 0] = 1
        Dy[-1, -1] = 1
        Iy = np.diag(np.ones(nx))
        L = np.kron(Ix, Dx) + np.kron(Dy, Iy)
        if not self.boundary:
            L[0,0] = 1.0
            return L
        else:
            L[0,0] = 1.0
            L[0,1:] = 0.0

            C = np.zeros((2**self.num_qubits,2**self.num_qubits))
            C[2**(self.num_qubits-1):,:2**(self.num_qubits-1)] = L.conj().T
            C[:2**(self.num_qubits-1),2**(self.num_qubits-1):] = L

            return C

    def eigs_bounds(self) -> Tuple[float, float]:
        """Return lower and upper bounds on the eigenvalues of the matrix."""
        lambdas = sorted(np.linalg.eigvals(self.matrix))
        return lambdas[0], lambdas[-1]

    def condition_bounds(self) -> Tuple[float, float]:
        """Return lower and upper bounds on the condition number of the matrix."""
        matrix_array = self.matrix
        kappa = np.linalg.cond(matrix_array)
        return kappa, kappa

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        """Check if the current configuration is valid."""
        valid = True

        if self.trotter_steps < 1:
            valid = False
            if raise_on_failure:
                raise AttributeError("The number of trotter steps should be a positive integer.")
            return False
        
        if self.trotter_order < 1 or (self.trotter_order > 1 and self.trotter_order % 2):
            valid = False
            if raise_on_failure:
                raise AttributeError("The trotter order must be 1 or a positive even integer.")
            return False

        return valid

    def _reset_registers(self, num_state_qubits: int) -> None:
        """Reset the quantum registers.

        Args:
            num_state_qubits: The number of qubits to represent the matrix.
        """
        qr_state = QuantumRegister(num_state_qubits, "state")
        self.qregs = [qr_state]

    def _build(self) -> None:
        """If not already built, build the circuit."""
        if self._is_built:
            return

        super()._build()

        self.compose(self.power(1), inplace=True)

    def inverse(self):
        return DiscreteLaplacian(
            nx=self.nx,
            ny=self.ny,
            boundary=self.boundary,
            trotterized=self.trotterized,
            tolerance=self.tolerance,
            evolution_time=-1 * self.evolution_time,
            trotter_order=self.trotter_order,
            trotter_steps=self.trotter_steps
        )

    def power(self, power: int) -> QuantumCircuit:
        """Build powers of the circuit.

        Args:
            power: The power to raise this circuit to.
            matrix_power: If True, the circuit is converted to a matrix and then the
                matrix power is computed. If False, and ``power`` is a positive integer,
                the implementation defaults to ``repeat``.

        Returns:
            The quantum circuit implementing powers of the unitary.
        """
        if self.trotterized:
            L = SparsePauliOp.from_operator(Operator(self.matrix)).simplify()

            evolution = PauliEvolutionGate(L, time=self.evolution_time)
            if self.trotter_order > 1:
                stevolution = SuzukiTrotter(order=self.trotter_order,reps=self.trotter_steps).synthesize(evolution)
                return stevolution
            else:
                stevolution = LieTrotter(reps=self.trotter_steps).synthesize(evolution)
                return stevolution
        else:
            qc = QuantumCircuit(self.num_state_qubits)
            evolved = sp.linalg.expm(-1j * self.matrix * self.evolution_time)
            qc.unitary(evolved, qc.qubits)
            return qc.power(power)
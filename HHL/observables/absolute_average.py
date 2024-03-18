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

# Modified 2/29/2024 by Robert Deaton

"""The absolute value of the average of a linear system of equations solution."""

from typing import Union, List
import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp

from .linear_system_observable import LinearSystemObservable


class AbsoluteAverage(LinearSystemObservable):
    r"""An observable for the absolute average of a linear system of equations solution.

    For a vector :math:`x=(x_1,...,x_N)`, the absolute average is defined as
    :math:`\abs{\frac{1}{N}\sum_{i=1}^{N}x_i}`.

    Examples:

        .. jupyter-execute::

            import numpy as np
            from qiskit import QuantumCircuit
            from qiskit.primitives import Estimator
            from qiskit.circuit.library import Isometry
            from linear_solvers.observables.absolute_average import AbsoluteAverage

            observable = AbsoluteAverage()
            vector = [1.0, -2.1, 3.2, -4.3]

            init_state = vector / np.linalg.norm(vector)
            num_qubits = int(np.log2(len(vector)))

            qc = QuantumCircuit(num_qubits)
            qc.append(Isometry(init_state, 0, 0), qargs=qc.qregs[0])
            qc.append(observable.observable_circuit(num_qubits), list(range(num_qubits)))

            # Observable operator
            estimator = Estimator()
            observable_op = observable.observable(num_qubits)
            state_vec = estimator.run(qc, observable_op).result().values

            # Obtain result
            result = observable.post_processing(state_vec, num_qubits)

            # Obtain analytical evaluation
            exact = observable.evaluate_classically(init_state)
    """

    def observable(self, num_qubits: int) -> SparsePauliOp:
        """The observable operator.

        Args:
            num_qubits: The number of qubits on which the observable will be applied.

        Returns:
            The observable as a sum of Pauli strings.
        """
        ####### generates zero_op ^{\tensor n}, since we can't use TensoredOp

        oplist = ['I','Z']
        if num_qubits == 0:
            tensored = SparsePauliOp('')
        else:
            for _ in range(num_qubits-1):
                oplist = ['I' + op for op in oplist] + ['Z' + op for op in oplist]
        tensored = 2**(-num_qubits) * SparsePauliOp(oplist)

        #######
        
        return tensored

    def observable_circuit(self, num_qubits: int) -> Union[QuantumCircuit, List[QuantumCircuit]]:
        """The circuit implementing the absolute average observable.

        Args:
            num_qubits: The number of qubits on which the observable will be applied.

        Returns:
            The observable as a QuantumCircuit.
        """
        qc = QuantumCircuit(num_qubits)
        qc.h(qc.qubits)
        return qc

    def post_processing(
        self, solution: Union[float, List[float]], num_qubits: int, scaling: float = 1
    ) -> float:
        """Evaluates the absolute average on the solution to the linear system.

        Args:
            solution: The probability calculated from the circuit and the observable.
            num_qubits: The number of qubits where the observable was applied.
            scaling: Scaling of the solution.

        Returns:
            The value of the absolute average.

        Raises:
            ValueError: If the input is not in the correct format.
        """
        if isinstance(solution, list):
            if len(solution) == 1:
                solution = solution[0]
            else:
                raise ValueError("Solution probability must be given as a single value.")

        return np.real(np.sqrt(solution / (2**num_qubits)) / scaling)

    def evaluate_classically(self, solution: Union[np.array, QuantumCircuit]) -> float:
        """Evaluates the given observable on the solution to the linear system.

        Args:
            solution: The solution to the system as a numpy array or the circuit that prepares it.

        Returns:
            The value of the observable.
        """
        # Check if it is QuantumCircuits and get the array from them
        if isinstance(solution, QuantumCircuit):
            solution = Statevector(solution).data
        return np.abs(np.mean(solution))
# Variational Quantum Linear Solver
# Ref :
# Tutorial :


"""Variational Quantum Linear Solver

See https://arxiv.org/abs/1909.05820
"""


from dataclasses import dataclass
from typing import Optional, Union, List, Callable, Dict, Tuple
import numpy as np

from qiskit import QuantumCircuit
from qiskit.primitives import BaseEstimator, BaseSampler
from qiskit.algorithms.variational_algorithm import VariationalAlgorithm
from qiskit.utils.validation import validate_min
from qiskit.algorithms.minimum_eigen_solvers.vqe import (
    _validate_bounds,
    _validate_initial_point,
)
from qiskit.circuit.library.n_local.real_amplitudes import RealAmplitudes

from qiskit.algorithms.optimizers import Minimizer, Optimizer
from qiskit.opflow.gradients import GradientBase

from variational_linear_solver import (
    VariationalLinearSolver,
    VariationalLinearSolverResult,
)
from matrix_decomposition import (
    SymmetricDecomposition,
    MatrixDecomposition,
    PauliDecomposition,
)
from hadamard_test import HadammardTest, HadammardOverlapTest


@dataclass
class VQLSLog:
    values: List
    parameters: List

    def update(self, count, cost, parameters):
        self.values.append(cost)
        self.parameters.append(parameters)
        print(f"VQLS Iteration {count} Cost {cost}", end="\r", flush=True)


class VQLS(VariationalAlgorithm, VariationalLinearSolver):
    r"""Systems of linear equations arise naturally in many real-life applications in a wide range
    of areas, such as in the solution of Partial Differential Equations, the calibration of
    financial models, fluid simulation or numerical field calculation. The problem can be defined
    as, given a matrix :math:`A\in\mathbb{C}^{N\times N}` and a vector
    :math:`\vec{b}\in\mathbb{C}^{N}`, find :math:`\vec{x}\in\mathbb{C}^{N}` satisfying
    :math:`A\vec{x}=\vec{b}`.

    Examples:

        .. jupyter-execute:

            from qalcore.qiskit.vqls.vqls import VQLS, VQLSLog
            from qiskit.circuit.library.n_local.real_amplitudes import RealAmplitudes
            from qiskit.algorithms import optimizers as opt
            from qiskit import Aer, BasicAer
            import numpy as np

            from qiskit.quantum_info import Statevector
            import matplotlib.pyplot as plt
            from qiskit.primitives import Estimator, Sampler, BackendEstimator

            # create random symmetric matrix
            A = np.random.rand(4, 4)
            A = A + A.T

            # create rhight hand side
            b = np.random.rand(4)

            # solve using numpy
            classical_solution = np.linalg.solve(A, b / np.linalg.norm(b))
            ref_solution = classical_solution / np.linalg.norm(classical_solution)

            # define the wave function ansatz
            ansatz = RealAmplitudes(2, entanglement="full", reps=3, insert_barriers=False)

            # define backend
            backend = BasicAer.get_backend("statevector_simulator")

            # define an estimator primitive
            estimator = Estimator()

            # define the logger
            log = VQLSLog([],[])

            # create the solver
            vqls = VQLS(
                estimator,
                ansatz,
                opt.CG(maxiter=200),
                callback=log.update
            )

            # solve
            res = vqls.solve(A, b, opt)
            vqls_solution = np.real(Statevector(res.state).data)

            # plot solution
            plt.scatter(ref_solution, vqls_solution)
            plt.plot([-1, 1], [-1, 1], "--")
            plt.show()

            # plot cost function
            plt.plot(log.values)
            plt.ylabel('Cost Function')
            plt.xlabel('Iterations')
            plt.show()

    References:

        [1] Carlos Bravo-Prieto, Ryan LaRose, M. Cerezo, Yigit Subasi, Lukasz Cincio,
        Patrick J. Coles. Variational Quantum Linear Solver
        `arXiv:1909.05820 <https://arxiv.org/abs/1909.05820>`
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        ansatz: QuantumCircuit,
        optimizer: Union[Optimizer, Minimizer],
        sampler: Optional[Union[BaseSampler, None]] = None,
        initial_point: Optional[Union[np.ndarray, None]] = None,
        gradient: Optional[Union[GradientBase, Callable, None]] = None,
        max_evals_grouped: Optional[int] = 1,
        callback: Optional[Callable[[int, np.ndarray, float, float], None]] = None,
    ) -> None:
        r"""
        Args:
            estimator: an Estimator primitive to compute the expected values of the
                quantum circuits needed for the cost function
            ansatz: A parameterized circuit used as Ansatz for the wave function.
            optimizer: A classical optimizer. Can either be a Qiskit optimizer or a callable
                that takes an array as input and returns a Qiskit or SciPy optimization result.
            sampler: a Sampler primitive to sample the output of some quantum circuits needed to
                compute the cost function. This is only needed if overal Hadammard tests are used.
            initial_point: An optional initial point (i.e. initial parameter values)
                for the optimizer. If ``None`` then VQLS will look to the ansatz for a preferred
                point and if not will simply compute a random one.
            gradient: An optional gradient function or operator for optimizer.
            max_evals_grouped: Max number of evaluations performed simultaneously. Signals the
                given optimizer that more than one set of parameters can be supplied so that
                potentially the expectation values can be computed in parallel. Typically this is
                possible when a finite difference gradient is used by the optimizer such that
                multiple points to compute the gradient can be passed and if computed in parallel
                improve overall execution time. Deprecated if a gradient operator or function is
                given.
            callback: a callback that can access the intermediate data during the optimization.
                Three parameter values are passed to the callback as follows during each evaluation
                by the optimizer for its current set of parameters as it works towards the minimum.
                These are: the evaluation count, the cost and the parameters for the ansatz
        """
        super().__init__()

        validate_min("max_evals_grouped", max_evals_grouped, 1)

        self._num_qubits = None
        self._max_evals_grouped = max_evals_grouped

        self.estimator = estimator
        self.sampler = sampler
        self.ansatz = ansatz
        self.optimizer = optimizer
        self.initial_point = initial_point

        self._gradient = None
        self.gradient = gradient

        self.callback = callback

        self._eval_count = 0

        self.vector_circuit = QuantumCircuit(0)
        self.matrix_circuits = QuantumCircuit(0)

        self.default_solve_options = {
            "use_overlap_test": False,
            "use_local_cost_function": False,
            "matrix_decomposition": "symmetric",
        }

    @property
    def num_qubits(self) -> int:
        """return the numner of qubits"""
        return self._num_qubits

    @num_qubits.setter
    def num_qubits(self, num_qubits: int) -> None:
        """Set the number of qubits"""
        self._num_qubits = num_qubits

    @property
    def num_clbits(self) -> int:
        """return the numner of classical bits"""
        return self._num_clbits

    @num_clbits.setter
    def num_clbits(self, num_clbits: int) -> None:
        """Set the number of classical bits"""
        self._num_clbits = num_clbits

    @property
    def ansatz(self) -> QuantumCircuit:
        """Returns the ansatz."""
        return self._ansatz

    @ansatz.setter
    def ansatz(self, ansatz: Optional[QuantumCircuit]):
        """Sets the ansatz.

        Args:
            ansatz: The parameterized circuit used as an ansatz.
            If None is passed, RealAmplitudes is used by default.

        """
        if ansatz is None:
            ansatz = RealAmplitudes()

        self._ansatz = ansatz
        self.num_qubits = ansatz.num_qubits + 1

    @property
    def initial_point(self) -> Union[np.ndarray, None]:
        """Returns initial point"""
        return self._initial_point

    @initial_point.setter
    def initial_point(self, initial_point: Union[np.ndarray, None]):
        """Sets initial point"""
        self._initial_point = initial_point

    @property
    def max_evals_grouped(self) -> int:
        """Returns max_evals_grouped"""
        return self._max_evals_grouped

    @max_evals_grouped.setter
    def max_evals_grouped(self, max_evals_grouped: int):
        """Sets max_evals_grouped"""
        self._max_evals_grouped = max_evals_grouped
        self.optimizer.set_max_evals_grouped(max_evals_grouped)

    @property
    def callback(self) -> Optional[Callable[[int, np.ndarray, float, float], None]]:
        """Returns callback"""
        return self._callback

    @callback.setter
    def callback(
        self, callback: Optional[Callable[[int, np.ndarray, float, float], None]]
    ):
        """Sets callback"""
        self._callback = callback

    @property
    def optimizer(self) -> Optimizer:
        """Returns optimizer"""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Optional[Optimizer]):
        """Sets the optimizer attribute.

        Args:
            optimizer: The optimizer to be used.

        """

        if isinstance(optimizer, Optimizer):
            optimizer.set_max_evals_grouped(self.max_evals_grouped)

        self._optimizer = optimizer

    def construct_circuit(
        self,
        matrix: Union[np.ndarray, QuantumCircuit, List],
        vector: Union[np.ndarray, QuantumCircuit],
        options: Dict,
    ) -> Tuple[List[QuantumCircuit], List[QuantumCircuit]]:
        """Returns the a list of circuits required to compute the expectation value

        Args:
            matrix (Union[np.ndarray, QuantumCircuit, List]): matrix of the linear system
            vector (Union[np.ndarray, QuantumCircuit]): rhs of thge linear system
            options (Dict): Options to compute define the quantum circuits
                that compute the cost function

        Raises:
            ValueError: if vector and matrix have different size
            ValueError: if vector and matrix have different number of qubits
            ValueError: the input matrix is not a numoy array nor a quantum circuit

        Returns:
            List[QuantumCircuit]: Quantum Circuits required to compute the cost function
        """

        # state preparation
        if isinstance(vector, QuantumCircuit):
            nqbit = vector.num_qubits
            self.vector_circuit = vector

        elif isinstance(vector, np.ndarray):
            # ensure the vector is double
            vector = vector.astype("float64")

            # create the circuit
            nqbit = int(np.log2(len(vector)))
            self.vector_circuit = QuantumCircuit(nqbit, name="Ub")

            # prep the vector if its norm is non nul
            vec_norm = np.linalg.norm(vector)
            if vec_norm != 0:
                self.vector_circuit.prepare_state(vector / vec_norm)
            else:
                raise ValueError("Norm of b vector is null!")

        # general numpy matrix
        if isinstance(matrix, np.ndarray):
            # ensure the matrix is double
            matrix = matrix.astype("float64")

            if matrix.shape[0] != 2**self.vector_circuit.num_qubits:
                raise ValueError(
                    "Input vector dimension does not match input "
                    "matrix dimension! Vector dimension: "
                    + str(self.vector_circuit.num_qubits)
                    + ". Matrix dimension: "
                    + str(matrix.shape[0])
                )
            decomposition = {
                "pauli": PauliDecomposition,
                "symmetric": SymmetricDecomposition,
            }[options["matrix_decomposition"]]
            self.matrix_circuits = decomposition(matrix=matrix)

        # a single circuit
        elif isinstance(matrix, QuantumCircuit):
            if matrix.num_qubits != self.vector_circuit.num_qubits:
                raise ValueError(
                    "Matrix and vector circuits have different numbers of qubits."
                )
            self.matrix_circuits = MatrixDecomposition(circuits=matrix)

        # if its a list of (coefficients, circuits)
        elif isinstance(matrix, List):
            assert isinstance(matrix[0][0], (float, complex))
            assert isinstance(matrix[0][1], QuantumCircuit)
            self.matrix_circuits = MatrixDecomposition(
                circuits=[m[1] for m in matrix], coefficients=[m[0] for m in matrix]
            )

        else:
            raise ValueError("Format of the input matrix not recognized")

        # create only the circuit for <psi|psi> =  <0|V A_n ^* A_m V|0>
        # with n != m as the diagonal terms (n==m) always give a proba of 1.0
        hdmr_tests_norm = self._get_norm_circuits()

        # create the circuits for <b|psi>
        # local cost function
        if options["use_local_cost_function"]:
            hdmr_tests_overlap = self._get_local_circuits()

        # global cost function
        else:
            hdmr_tests_overlap = self._get_global_circuits(options)

        return hdmr_tests_norm, hdmr_tests_overlap

    def _get_norm_circuits(self) -> List[QuantumCircuit]:
        """construct the circuit for the norm

        Returns:
            List[QuantumCircuit]: quantum circuits needed for the norm
        """

        hdmr_tests_norm = []

        for ii_mat in range(len(self.matrix_circuits)):
            mat_i = self.matrix_circuits[ii_mat]

            for jj_mat in range(ii_mat + 1, len(self.matrix_circuits)):
                mat_j = self.matrix_circuits[jj_mat]
                hdmr_tests_norm.append(
                    HadammardTest(
                        operators=[mat_i.circuit.inverse(), mat_j.circuit],
                        apply_initial_state=self._ansatz,
                        apply_measurement=False,
                    )
                )
        return hdmr_tests_norm

    def _get_local_circuits(self) -> List[QuantumCircuit]:
        """construct the circuits needed for the local cost function

        Returns:
            List[QuantumCircuit]: quantum circuits for the local cost function
        """

        hdmr_tests_overlap = []
        num_z = self.matrix_circuits[0].circuit.num_qubits

        # create the circuits for <0| U^* A_l V(Zj . Ij|) V^* Am^* U|0>
        for ii_mat in range(len(self.matrix_circuits)):
            mat_i = self.matrix_circuits[ii_mat]

            for jj_mat in range(ii_mat, len(self.matrix_circuits)):
                mat_j = self.matrix_circuits[jj_mat]

                for iqubit in range(num_z):
                    # circuit for the CZ operation on the iqth qubit
                    qc_z = QuantumCircuit(num_z + 1)
                    qc_z.cz(0, iqubit + 1)

                    # create Hadammard circuit
                    hdmr_tests_overlap.append(
                        HadammardTest(
                            operators=[
                                mat_i.circuit,
                                self.vector_circuit.inverse(),
                                qc_z,
                                self.vector_circuit,
                                mat_j.circuit.inverse(),
                            ],
                            apply_control_to_operator=[True, True, False, True, True],
                            apply_initial_state=self.ansatz,
                            apply_measurement=False,
                        )
                    )
        return hdmr_tests_overlap

    def _get_global_circuits(self, options: dict) -> List[QuantumCircuit]:
        """construct circuits needed for the global cost function

        Args:
            options (Dict): Options to define the quantum circuits that compute
                the cost function

        Returns:
            List[QuantumCircuit]: quantum circuits needed for the global cost function
        """

        # create the circuits for <0|U^* A_l V|0\rangle\langle 0| V^* Am^* U|0>
        # either using overal test or hadammard test
        if options["use_overlap_test"]:
            hdmr_overlap_tests = []
            for ii_mat in range(len(self.matrix_circuits)):
                mat_i = self.matrix_circuits[ii_mat]

                for jj_mat in range(ii_mat, len(self.matrix_circuits)):
                    mat_j = self.matrix_circuits[jj_mat]

                    hdmr_overlap_tests.append(
                        HadammardOverlapTest(
                            operators=[
                                self.vector_circuit,
                                mat_i.circuit,
                                mat_j.circuit,
                            ],
                            apply_initial_state=self.ansatz,
                            apply_measurement=True,
                        )
                    )
            return hdmr_overlap_tests

        # or using the normal Hadamard tests
        hdmr_tests = []
        for mat_i in self.matrix_circuits:
            hdmr_tests.append(
                HadammardTest(
                    operators=[
                        self.ansatz,
                        mat_i.circuit,
                        self.vector_circuit.inverse(),
                    ],
                    apply_measurement=False,
                )
            )

        return hdmr_tests

    @staticmethod
    def get_coefficient_matrix(coeffs) -> np.ndarray:
        """Compute all the vi* vj terms

        Args:
            coeffs (np.ndarray): list of complex coefficients
        """
        return coeffs[:, None].conj() @ coeffs[None, :]

    def _assemble_cost_function(
        self,
        hdmr_values_norm: np.ndarray,
        hdmr_values_overlap: np.ndarray,
        coefficient_matrix: np.ndarray,
        options: Dict,
    ) -> float:
        """Computes the value of the cost function

        Args:
            hdmr_values_norm (np.ndarray): values of the hadamard test for the norm
            hdmr_values_overlap (np.ndarray): values of the hadamard tests for the overlap
            coefficient_matrix (np.ndarray): exapnsion coefficients of the matrix
            options (Dict): options to compute cost function

        Returns:
            float: value of the cost function
        """

        # compute all the terms in <\phi|\phi> = \sum c_i* cj <0|V Ai* Aj V|0>
        norm = self._compute_normalization_term(coefficient_matrix, hdmr_values_norm)

        if options["use_local_cost_function"]:
            # compute all terms in
            # \sum c_i* c_j 1/n \sum_n <0|V* Ai U Zn U* Aj* V|0>
            sum_terms = self._compute_local_terms(
                coefficient_matrix, hdmr_values_overlap, norm
            )

        else:
            # compute all the terms in
            # |<b|\phi>|^2 = \sum c_i* cj <0|U* Ai V|0><0|V* Aj* U|0>
            sum_terms = self._compute_global_terms(
                coefficient_matrix, hdmr_values_overlap, options
            )

        # overall cost
        cost = 1.0 - np.real(sum_terms / norm)

        # print("Cost function %f" % cost)
        return cost

    def _compute_normalization_term(
        self,
        coeff_matrix: np.ndarray,
        hdmr_values: np.ndarray,
    ) -> float:
        """Compute <phi|phi>

        .. math::
            \\langle\\Phi|\\Phi\\rangle = \\sum_{nm} c_n^*c_m \\langle 0|V^* U_n^* U_m V|0\\rangle

        Args:
            coeff_matrix (List): the matrix values of the c_n^* c_m coefficients
            hdmr_values (List): the values of the circuits output

        Returns:
            float: value of the sum
        """

        # compute all the terms in <\phi|\phi> = \sum c_i* cj <0|V Ai* Aj V|0>
        # hdrm_values here contains the values of the <0|V Ai* Aj V|0>  with j>i
        out = hdmr_values

        # we multiuply hdmrval by the triup coeff matrix and sum
        out *= coeff_matrix[np.triu_indices_from(coeff_matrix, k=1)]
        out = out.sum()

        # add the conj that corresponds to the tri down matrix
        out += out.conj()

        # add the diagonal terms
        # since <0|V Ai* Aj V|0> = 1 we simply
        # add the sum of the cici coeffs
        out += np.trace(coeff_matrix)

        return out.item()

    def _compute_global_terms(
        self, coeff_matrix: np.ndarray, hdmr_values: np.ndarray, options: Dict
    ) -> float:
        """Compute |<b|phi>|^2

        .. math::
            |\\langle b|\\Phi\\rangle|^2 = \\sum_{nm} c_n^*c_m \\gamma_{nm}

        with

        .. math::

            \\gamma_nm = \\langle 0|V^* U_n^* U_b |0 \\rangle \\langle 0|U_b^* U_m V |0\\rangle

        Args:
            coeff_matrix (np.ndarray): the matrix values of the c_n^* c_m coefficients
            hdmr_values (List): values of the circuit outputs
            options (Dict): options to compute cost function

        Returns:
            float: value of the sum
        """

        if options["use_overlap_test"]:
            # hdmr_values here contains the values of <0|V* Ai* U|0><0|V Aj U|0> for j>=i
            # we first insert these values in a tri up matrix
            size = len(self.matrix_circuits)
            hdmr_matrix = np.zeros((size, size)).astype("complex128")
            hdmr_matrix[np.tril_indices(size)] = hdmr_values

            # add the conj that correspond to the tri low part of the matrix
            # warning the diagonal is also contained in out and we only
            # want to add the conj of the tri up excluding the diag
            hdmr_matrix[np.triu_indices_from(hdmr_matrix, k=1)] = hdmr_matrix[
                np.tril_indices_from(hdmr_matrix, k=-1)
            ].conj()

            # multiply by the coefficent matrix and sum the values
            out_matrix = coeff_matrix * hdmr_matrix
            out = out_matrix.sum()

        else:
            # hdmr_values here contains the values of <0|V* Ai* U|0>
            # compute the matrix of the <0|V* Ai* U|0> <0|V Aj U*|0> values
            hdmr_matrix = self.get_coefficient_matrix(hdmr_values)
            out = (coeff_matrix * hdmr_matrix).sum()

        return out

    def _compute_local_terms(
        self, coeff_matrix: np.ndarray, hdmr_values: np.ndarray, norm: float
    ) -> float:
        """Compute the term of the local cost function given by

        .. math::
            \\sum c_i^* c_j \\frac{1}{n} \\sum_n \\langle 0|V^* A_i U Z_n U^* A_j^* V|0\\rangle

        Args:
            coeff_matrix (np.ndarray): the matrix values of the c_n^* c_m coefficients
            hdmr_values (List): values of the circuit outputs
            norm (float): value of the norm term

        Returns:
            float: value of the sum
        """

        # add all the hadamard test values corresponding to the insertion
        # of Z gates on the same cicuit
        # b_ij = \sum_n \\frac{1}{n} \\sum_n \\langle 0|V^* A_i U Z_n U^* A_j^* V|0\\rangle
        num_zgate = self.matrix_circuits[0].circuit.num_qubits
        hdmr_values = hdmr_values.reshape(-1, num_zgate).mean(1)

        # hdmr_values then contains the values of <0|V* Ai* U|0><0|V Aj U|0> for j>=i
        # we first insert these values in a tri up matrix
        size = len(self.matrix_circuits)
        hdmr_matrix = np.zeros((size, size)).astype("complex128")
        hdmr_matrix[np.triu_indices(size)] = hdmr_values

        # add the conj that correspond to the tri low part of the matrix
        # warning the diagonal is also contained in out and we only
        # want to add the conj of the tri up excluding the diag
        hdmr_matrix[np.tril_indices_from(hdmr_matrix, k=-1)] = hdmr_matrix[
            np.triu_indices_from(hdmr_matrix, k=1)
        ].conj()

        # multiply by the coefficent matrix and sum the values
        out_matrix = coeff_matrix * hdmr_matrix
        out = (out_matrix).sum()

        # add \sum c_i* cj <0|V Ai* Aj V|0>
        out += norm

        # factor two coming from |0><0| = 1/2(I+Z)
        out /= 2

        return out

    def get_cost_evaluation_function(
        self,
        hdmr_tests_norm: List,
        hdmr_tests_overlap: List,
        coefficient_matrix: np.ndarray,
        options: Dict,
    ) -> Callable[[np.ndarray], Union[float, List[float]]]:
        """Generate the cost function of the minimazation process

        Args:
            hdmr_tests_norm (List): list of quantum circuits needed to compute the norm
            hdmr_tests_overlap (List): list of quantum circuits needed to compute the norm
            coefficient_matrix (np.ndarray): the matrix values of the c_n^* c_m coefficients
            options (Dict): Option to compute the cost function

        Raises:
            RuntimeError: If the ansatz is not parametrizable

        Returns:
            Callable[[np.ndarray], Union[float, List[float]]]: the cost function
        """

        num_parameters = self.ansatz.num_parameters
        if num_parameters == 0:
            raise RuntimeError(
                "The ansatz must be parameterized, but has 0 free parameters."
            )

        def cost_evaluation(parameters):
            # estimate the expected values of the norm circuits
            hdmr_values_norm = np.array(
                [hdrm.get_value(self.estimator, parameters) for hdrm in hdmr_tests_norm]
            )

            if options["use_overlap_test"]:
                hdmr_values_overlap = np.array(
                    [
                        hdrm.get_value(self.sampler, parameters)
                        for hdrm in hdmr_tests_overlap
                    ]
                )
            else:
                hdmr_values_overlap = np.array(
                    [
                        hdrm.get_value(self.estimator, parameters)
                        for hdrm in hdmr_tests_overlap
                    ]
                )
            # compute the total cost
            cost = self._assemble_cost_function(
                hdmr_values_norm, hdmr_values_overlap, coefficient_matrix, options
            )

            # get the intermediate results if required
            if self._callback is not None:
                self._eval_count += 1
                self._callback(self._eval_count, cost, parameters)
            else:
                self._eval_count += 1
                print(
                    f"VQLS Iteration {self._eval_count} Cost {cost}",
                    end="\r",
                    flush=True,
                )

            return cost

        return cost_evaluation

    def _validate_solve_options(self, options: Union[Dict, None]) -> Dict:
        """validate the options used for the solve methods

        Args:
            options (Union[Dict, None]): options
        """
        valid_keys = self.default_solve_options.keys()

        if options is None:
            options = self.default_solve_options

        else:
            for k in options.keys():
                if k not in valid_keys:
                    raise ValueError(
                        "Option {k} not recognized, valid keys are {valid_keys}"
                    )
            for k in valid_keys:
                if k not in options.keys():
                    options[k] = self.default_solve_options[k]

        if options["use_overlap_test"] and options["use_local_cost_function"]:
            raise ValueError(
                "Local cost function cannot be used with Hadamard Overlap test"
            )

        if options["use_overlap_test"] and self.sampler is None:
            raise ValueError(
                "Please provide a sampler primitives when using Hadamard Overlap test"
            )

        valid_matrix_decomposition = ["symmetric", "pauli"]
        if options["matrix_decomposition"] not in valid_matrix_decomposition:
            raise ValueError(
                "matrix decomposition {k} not recognized, \
                    valid keys are {valid_matrix_decomposition}"
            )

        return options

    def solve(
        self,
        matrix: Union[np.ndarray, QuantumCircuit, List[QuantumCircuit]],
        vector: Union[np.ndarray, QuantumCircuit],
        options: Union[Dict, None] = None,
    ) -> VariationalLinearSolverResult:
        """Solve the linear system

        Args:
            matrix (Union[List, np.ndarray, QuantumCircuit]): matrix of the linear system
            vector (Union[np.ndarray, QuantumCircuit]): rhs of the linear system
            options (Union[Dict, None]): options for the calculation of the cost function

        Returns:
            VariationalLinearSolverResult: Result of the optimization
                and solution vector of the linear system
        """

        # validate the options
        options = self._validate_solve_options(options)

        # compute the circuits needed for the hadamard tests
        hdmr_tests_norm, hdmr_tests_overlap = self.construct_circuit(
            matrix, vector, options
        )

        # compute he coefficient matrix
        coefficient_matrix = self.get_coefficient_matrix(
            np.array([mat_i.coeff for mat_i in self.matrix_circuits])
        )

        # set an expectation for this algorithm run (will be reset to None at the end)
        initial_point = _validate_initial_point(self.initial_point, self.ansatz)
        bounds = _validate_bounds(self.ansatz)

        # Convert the gradient operator into a callable function that is compatible with the
        # optimization routine.
        gradient = self._gradient
        self._eval_count = 0

        # get the cost evaluation function
        cost_evaluation = self.get_cost_evaluation_function(
            hdmr_tests_norm, hdmr_tests_overlap, coefficient_matrix, options
        )

        if callable(self.optimizer):
            opt_result = self.optimizer(  # pylint: disable=not-callable
                fun=cost_evaluation, x0=initial_point, jac=gradient, bounds=bounds
            )
        else:
            opt_result = self.optimizer.minimize(
                fun=cost_evaluation, x0=initial_point, jac=gradient, bounds=bounds
            )

        # create the solution
        solution = VariationalLinearSolverResult()

        # optimization data
        solution.optimal_point = opt_result.x
        solution.optimal_parameters = dict(zip(self.ansatz.parameters, opt_result.x))
        solution.optimal_value = opt_result.fun
        solution.cost_function_evals = opt_result.nfev

        # final ansatz
        solution.state = self.ansatz.assign_parameters(solution.optimal_parameters)

        return solution

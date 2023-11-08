from qiskit.quantum_info.operators import Operator, SparsePauliOp
from qiskit.quantum_info import Pauli
from scipy import sparse
import numpy as np

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
def repeated_x(dim):
    if dim ==0:
        return 1
    else:
        term = X
        for i in range(dim-1):
            term = term^X
        return term
def laplace_paulis(dim: int):
    """
	Construct 1-D Laplacian operator in terms of closed form Paulis
	dim (int) : Number of qubits used to define the size of the Laplacian
	return (SparsePauliOp): Qiskit operator in terms of a sum of Pauli operators
	"""
    if dim ==0:
        return 2
    elif dim ==1:
        return 2*Ident(dim)- repeated_x(dim)
    elif dim ==2:
        return 2*Ident(dim) - (Ident(dim-1)^repeated_x(dim-1))-(1/2*repeated_x(dim)+1/2*(Y^Y))
    else:
        sum_term = 0
        for i in range(0,dim-1):
            if i == 0:
                sum_term +=1/(2**(i+2))*repeated_x(i)*(laplace_paulis(dim-2-i)^Y^Y)
            else:
                if (dim-2-i)==0:
                    sum_term +=1/(2**(i+2))*laplace_paulis(dim-2-i)*Y^repeated_x(i)^Y
                else:
                    sum_term +=1/(2**(i+2))*laplace_paulis(dim-2-i)^Y^repeated_x(i)^Y
        ans = 2*Ident(dim)+(1/2*(laplace_paulis(dim-1)-4*Ident(dim-1))^X)-sum_term
        return ans.simplify()       
def two_d_laplacian_paulis(dim:int):
    """
	Construct 2-D Laplacian operator from the 1-D Laplacian already in terms of Paulis
	dim (int) : Number of qubits used to define the size of the Laplacian
	return (SparsePauliOp): Qiskit operator in terms of a sum of Pauli operators
	"""
    return (Ident(dim)^laplace_paulis(dim))+(laplace_paulis(dim)^Ident(dim))
    
def calculate_laplace_three_point(bound1,bound2,size):
         
        main_diag = 2.0 * np.ones(size)
        main_diag[0] = bound1
        main_diag[-1]= bound2
        off_diag  =  -1.0 * np.ones(size-1)
        laplace_term = sparse.diags([main_diag, off_diag, off_diag], (0, -1, 1))
        return laplace_term
    
def form_two_d_poisson(nx,ny):
    spx = sparse.kron(sparse.eye(nx),calculate_laplace_three_point(2,2,ny),format='csr')
    spy = sparse.kron(calculate_laplace_three_point(2,2,nx),sparse.eye(ny),format='csr')
    return spx+spy
def Laplacian(nx, ny):
    Dx = np.diag(np.ones(nx)) * 2 - np.diag(np.ones(nx - 1), 1) - np.diag(np.ones(nx - 1), -1)
    Dx[0, 0] = 1
    Dx[-1, -1] = 1
    Ix = np.diag(np.ones(ny))
    Dy = np.diag(np.ones(ny)) * 2 - np.diag(np.ones(ny - 1), 1) - np.diag(np.ones(ny - 1), -1)
    Dy[0, 0] = 1
    Dy[-1, -1] = 1
    Iy = np.diag(np.ones(nx))
    L = sparse.kron(Ix, Dx,format='csr') + sparse.kron(Dy, Iy,format='csr')
    return L
def Laplacian_d(nx, ny):
    Dx = np.diag(np.ones(nx)) * 2 - np.diag(np.ones(nx - 1), 1) - np.diag(np.ones(nx - 1), -1)
    Dx[0, 0] = 1
    Dx[-1, -1] = 1
    Ix = np.diag(np.ones(ny))
    Dy = np.diag(np.ones(ny)) * 2 - np.diag(np.ones(ny - 1), 1) - np.diag(np.ones(ny - 1), -1)
    Dy[0, 0] = 1
    Dy[-1, -1] = 1
    Iy = np.diag(np.ones(nx))
    L = np.kron(Ix, Dx) + np.kron(Dy, Iy)
    return L
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
def convert_matrix_to_qiskit_paulis(matrix):
    paulis = []
    out_list = h2zixy(matrix).splitlines()
    for i in out_list:
        find_star = i.find('*')
        string = i[find_star+1:]
        coeff = float(i[:find_star])
        paulis.append((string,coeff))
    pauli = SparsePauliOp.from_list(paulis)
    return pauli
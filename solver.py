import numpy as np

def SOR(A, b, omega, initial_guess, max_iter=1000, tol=1e-6):
    n = A.shape[0]
    x = initial_guess.copy()
    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)
    
    for k in range(max_iter):
        old_x = x.copy()
        
        for i in range(n):
            sum1 = np.dot(A[i, :i], x[:i])
            sum2 = np.dot(A[i, i+1:], x[i+1:])
            x[i] = (1 - omega) * x[i] + omega * (b[i] - sum1 - sum2) / A[i, i]
        
        if np.linalg.norm(x - old_x) < tol:
            break
            
    return x








import numpy as np
import cvxpy as cp
from scipy.linalg import eigh

# lazy randn
def randn(*dims):
    return np.random.randn(*dims)

def rand(*dims):
    return np.random.rand(*dims)

def rand_qp(nx=10, n_eq=5, n_ineq=5, seed=1, compute_sol=True):
    np.random.seed(seed)
    H = randn(nx, nx)
    H = H.T @ H + np.eye(nx)
    H = H + H.T

    A = randn(n_eq, nx)
    C = randn(n_ineq, nx)

    active_ineq = randn(n_ineq) > 0.5

    mu = randn(n_eq)
    lamb = (randn(n_ineq))*active_ineq

    x = randn(nx)
    b = A@x
    d = C@x - randn(n_ineq)*(~active_ineq)

    g = -H@x - A.T@mu - C.T@lamb
    
    if compute_sol:
        x = cp.Variable(nx)
        prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, np.array(H)) + g.T@x), [A@x == b, C@x >= d])
        prob.solve()
        return (H, g, np.vstack((A, C)), np.concatenate((b, d)), 
            np.concatenate((b, np.full(n_ineq, np.inf))), x.value)
    else:
        return (H, g, np.vstack((A, C)), np.concatenate((b, d)), 
            np.concatenate((b, np.full(n_ineq, np.inf))), None)


def update_qp(H, A, n_eq, n_ineq, seed=1, compute_sol=True):
    """
    Update the QP problem with vectors
    """
    np.random.seed(seed)
    nx = H.shape[0]
    C = A[n_eq:]
    A = A[:n_eq]
    
    active_ineq = randn(n_ineq) > 0.5
    mu = randn(n_eq)
    lamb = (randn(n_ineq))*active_ineq

    x = randn(nx)
    b = A@x
    d = C@x - randn(n_ineq)*(~active_ineq)

    g = -H@x - A.T@mu - C.T@lamb

    if compute_sol:
        x = cp.Variable(nx)
        prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, np.array(H)) + g.T@x), [A@x == b, C@x >= d])
        prob.solve()

        return (H, g, np.vstack((A, C)), np.concatenate((b, d)),
            np.concatenate((b, np.full(n_ineq, np.inf))), x.value)
    else:
        return (H, g, np.vstack((A, C)), np.concatenate((b, d)),
            np.concatenate((b, np.full(n_ineq, np.inf))), None)

def qp_initialization(H: np.ndarray, A: np.ndarray, l: np.ndarray, u: np.ndarray):
    eq_tol = 1e-6
    nx = H.shape[0]
    nc = A.shape[0]

    # Decomposition for fast update of ADMM hessian
    eigs, U = eigh(H, driver="evd")
    e_inv = 1 / eigs
    e_sqrt = np.sqrt(e_inv)
    L_inv = U @ np.diag(e_sqrt) @ U.T
    H_inv = U @ np.diag(e_inv) @ U.T

    # Scaling matrix enforces higher penalty to equality constraints
    sqrtm = np.ones(nc)
    m = np.ones(nc)
    m_inv  = np.ones(nc)
    sqrtm[(u - l) <= eq_tol] = np.sqrt(1e3)
    m[(u - l) <= eq_tol] = 1e3
    m_inv[(u - l) <= eq_tol] = 1e-3
    sqrtM = np.diag(sqrtm)

    S = L_inv @ A.T @ sqrtM
    eigs, U = eigh(S @ S.T, driver="evd")
    T = L_inv.T @ U
    return T, eigs, m, m_inv
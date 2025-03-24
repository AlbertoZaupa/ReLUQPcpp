# distutils: language = c++

# We need to work with C++ std::vector.
from libcpp.vector cimport vector
from libcpp cimport bool
import numpy as np
cimport numpy as np

cdef extern from "solver.hpp" namespace "":
    # Forward-declare the Solver class. We only wrap the public members.
    cdef cppclass Solver:
        # Constructor
        Solver(float rho_, int nx, int nc,
               float* H_, float* A_, float* T_,
               float* M_, float* M_inv_,
               float* g_, float* l_, float* u_,
               float* eigs_)
        # Destructor
        # (Cython will call delete when we wrap it)
        void solve()
        void setup()
        void update(const float* g, const float* l, const float* u, float rho_)
        vector[float] get_results()
        float solve_time
        float update_time
        int iter

class SolverResults:
    def __init__(self, x_sol, solve_time, iter):
        self.x_sol = x_sol
        self.solve_time = solve_time
        self.iter = iter

cdef class PySolver:
    cdef Solver* solver_ptr

    def __cinit__(self, float rho, int nx, int nc,
                  np.ndarray H, np.ndarray A, np.ndarray T,
                  np.ndarray M, np.ndarray M_inv,
                  np.ndarray g, np.ndarray l, np.ndarray u, np.ndarray eigs):
        """
        Create a new Solver instance.
        The arrays H, A, T, M, M_inv, g, l, u, eigs should be provided as
        numpy arrays of dtype np.float32 or np.float64.
        """
        # Check if input arrays are either float32 or float64 and convert if necessary
        if H.dtype == np.float64:
            H = H.astype(np.float32)
        if A.dtype == np.float64:
            A = A.astype(np.float32)
        if T.dtype == np.float64:
            T = T.astype(np.float32)
        if M.dtype == np.float64:
            M = M.astype(np.float32)
        if M_inv.dtype == np.float64:
            M_inv = M_inv.astype(np.float32)
        if g.dtype == np.float64:
            g = g.astype(np.float32)
        if l.dtype == np.float64:
            l = l.astype(np.float32)
        if u.dtype == np.float64:
            u = u.astype(np.float32)
        if eigs.dtype == np.float64:
            eigs = eigs.astype(np.float32)

        # Check that the input arrays are contiguous (important for performance)
        if not H.flags['C_CONTIGUOUS'] or not A.flags['C_CONTIGUOUS'] or not T.flags['C_CONTIGUOUS'] or \
           not M.flags['C_CONTIGUOUS'] or not M_inv.flags['C_CONTIGUOUS'] or not g.flags['C_CONTIGUOUS'] or \
           not l.flags['C_CONTIGUOUS'] or not u.flags['C_CONTIGUOUS'] or not eigs.flags['C_CONTIGUOUS']:
            raise ValueError("Input arrays must be contiguous (C_CONTIGUOUS) numpy arrays.")

        # Comply with CUBLAS column-major order
        A = A.copy(order='F')
        T = T.copy(order='F')
        H = H.copy(order='F')

        # Create the Solver instance.
        self.solver_ptr = new Solver(rho, nx, nc,
                                     <float*>H.data,
                                     <float*>A.data,
                                     <float*>T.data,
                                     <float*>M.data,
                                     <float*>M_inv.data,
                                     <float*>g.data,
                                     <float*>l.data,
                                     <float*>u.data,
                                     <float*>eigs.data)

    def __dealloc__(self):
        if self.solver_ptr is not NULL:
            del self.solver_ptr

    def solve(self):
        self.solver_ptr.solve()

    def setup(self):
        self.solver_ptr.setup()

    def update(self, np.ndarray g, np.ndarray l, np.ndarray u, float rho):
        """
        Update the solver with new data.
        The arrays g, l, u must be numpy arrays of dtype np.float32 or np.float64.
        """
        # Convert to float32 if dtype is float64
        if g.dtype == np.float64:
            g = g.astype(np.float32)
        if l.dtype == np.float64:
            l = l.astype(np.float32)
        if u.dtype == np.float64:
            u = u.astype(np.float32)

        # Ensure that the arrays are contiguous
        if not g.flags['C_CONTIGUOUS'] or not l.flags['C_CONTIGUOUS'] or not u.flags['C_CONTIGUOUS']:
            raise ValueError("Input arrays must be contiguous numpy arrays.")

        # Update the solver with the new data
        self.solver_ptr.update(<float*>g.data, <float*>l.data, <float*>u.data, rho)

    def get_results(self):
        """
        Returns the results as a Python list of floats.
        """
        cdef vector[float] vec = self.solver_ptr.get_results()
        cdef int n = vec.size()
        py_list = [0.0] * n
        cdef int i
        for i in range(n):
            py_list[i] = vec[i]
        return SolverResults(np.array(py_list), self.solver_ptr.solve_time + self.solver_ptr.update_time, self.solver_ptr.iter)

    @property
    def solve_time(self):
        """
        Expose the public member solve_time.
        """
        return self.solver_ptr.solve_time

    @property
    def update_time(self):
        """
        Expose the public member solve_time.
        """
        return self.solver_ptr.update_time

    @property
    def iter(self):
        """
        Expose the public member iter.
        """
        return self.solver_ptr.iter

# distutils: language = c++

# We need to work with C++ std::vector.
from libcpp.vector cimport vector
from libcpp cimport bool
import numpy as np
cimport numpy as np

real_type = np.float32

ctypedef float real

cdef extern from "solver.hpp" namespace "":
    # Forward-declare the Solver class. We only wrap the public members.
    cdef cppclass Solver:
        # Constructor
        Solver(bool verbose, int reactive_rho_duration, real rho_, int nx, int nc,
               real* H_, real* A_, real* T_,
               real* M_, real* M_inv_,
               real* g_, real* l_, real* u_,
               real* eigs_)
        # Destructor
        # (Cython will call delete when we wrap it)
        void solve()
        void setup(real abs_tol, int max_iter, int check_interval)
        void update(const real* g, const real* l, const real* u, real rho_)
        vector[real] get_results()
        real solve_time
        real update_time
        int iter
        

class SolverResults:
    class Info:
        def __init__(self, x_sol, solve_time, iter):
            self.solve_time = solve_time
            self.iter = iter

    def __init__(self, x_sol, solve_time, iter):
        self.x = x_sol
        self.info = self.Info(x_sol, solve_time, iter)


cdef class GpuSolver:
    cdef Solver* solver_ptr

    def __cinit__(self, real rho, int nx, int nc,
                  np.ndarray H, np.ndarray A, np.ndarray T,
                  np.ndarray M, np.ndarray M_inv,
                  np.ndarray g, np.ndarray l, np.ndarray u, np.ndarray eigs, 
                  int reactive_rho_duration = 0, bool verbose = False):
        """
        Create a new Solver instance.
        The arrays H, A, g, l, u should be provided as
        numpy arrays of dtype np.float32 or np.float64.
        """
        # Check if input arrays are either float32 or float64 and convert if necessary
        H = H.astype(real_type)
        A = A.astype(real_type)
        g = g.astype(real_type)
        l = l.astype(real_type)
        u = u.astype(real_type)
        T = T.astype(real_type)
        eigs = eigs.astype(real_type)
        M = M.astype(real_type)
        M_inv = M_inv.astype(real_type)

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
        self.solver_ptr = new Solver(verbose, reactive_rho_duration, rho, nx, nc,
                                     <real*>H.data,
                                     <real*>A.data,
                                     <real*>T.data,
                                     <real*>M.data,
                                     <real*>M_inv.data,
                                     <real*>g.data,
                                     <real*>l.data,
                                     <real*>u.data,
                                     <real*>eigs.data)

    def __dealloc__(self):
        if self.solver_ptr is not NULL:
            del self.solver_ptr

    def solve(self):
        self.solver_ptr.solve()
        cdef vector[real] vec = self.solver_ptr.get_results()
        cdef int n = vec.size()
        py_list = [0.0] * n
        cdef int i
        for i in range(n):
            py_list[i] = vec[i]
        return SolverResults(np.array(py_list), self.solver_ptr.solve_time + self.solver_ptr.update_time, self.solver_ptr.iter)

    def setup(self, abs_tol=1e-3, max_iter=4000, check_interval=25):
        self.solver_ptr.setup(abs_tol, max_iter, check_interval)

    def update(self, np.ndarray g, np.ndarray l, np.ndarray u, real rho):
        """
        Update the solver with new data.
        The arrays g, l, u must be numpy arrays of dtype np.float32 or np.float64.
        """
        # Convert to float32 if dtype is float64
        g = g.astype(real_type)
        l = l.astype(real_type)
        u = u.astype(real_type)

        # Ensure that the arrays are contiguous
        if not g.flags['C_CONTIGUOUS'] or not l.flags['C_CONTIGUOUS'] or not u.flags['C_CONTIGUOUS']:
            raise ValueError("Input arrays must be contiguous numpy arrays.")

        # Update the solver with the new data
        self.solver_ptr.update(<real*>g.data, <real*>l.data, <real*>u.data, rho)

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

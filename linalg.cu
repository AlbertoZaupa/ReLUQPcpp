#include <cublas_v2.h>
#include "linalg.hpp"

// -----------------------------
// Matrix and Vector Classes
// -----------------------------

Matrix::Matrix() {
    rows = 0;
    cols = 0;
    d_data = nullptr;
}

Matrix::Matrix(int r, int c) {
    rows = r;
    cols = c;
    CUDA_CHECK(cudaMalloc(&d_data, rows * cols * sizeof(float)));
}

Matrix::~Matrix() {
    if (d_data) cudaFree(d_data);
}

// Copies data from host pointer to device memory
void Matrix::copyFromHost(const float* h_data) {
    CUDA_CHECK(cudaMemcpy(d_data, h_data, rows * cols * sizeof(float), cudaMemcpyHostToDevice));
}

// Copies data from device pointer
void Matrix::copyFromDevice(const float* d_data_) {
    CUDA_CHECK(cudaMemcpy(d_data, d_data_, rows * cols * sizeof(float), cudaMemcpyDeviceToDevice));
}

// Copies data from device memory to host pointer
void Matrix::copyToHost(float* h_data) const {
    CUDA_CHECK(cudaMemcpy(h_data, d_data, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));
}

Vector::Vector() {
    n_elements = 0;
    d_data = nullptr;
}

Vector::Vector(int n) : n_elements(n) {
    CUDA_CHECK(cudaMalloc(&d_data, n_elements * sizeof(float)));
}

Vector::~Vector() {
    if (d_data) cudaFree(d_data);
}

// Copies data from host pointer to device memory
void Vector::copyFromHost(const float* h_data) {
    CUDA_CHECK(cudaMemcpy(d_data, h_data, n_elements * sizeof(float), cudaMemcpyHostToDevice));
}

// Copies data from device pointer
void Vector::copyFromDevice(const float* d_data_) {
    CUDA_CHECK(cudaMemcpy(d_data, d_data_, n_elements * sizeof(float), cudaMemcpyDeviceToDevice));
}

// Copies data from device memory to host pointer
void Vector::copyToHost(float* h_data) const {
    CUDA_CHECK(cudaMemcpy(h_data, d_data, n_elements * sizeof(float), cudaMemcpyDeviceToHost));
}

// -----------------------------
// CUDA Kernels for custom operations
// -----------------------------

// Kernel to perform vector difference
__global__ void subKernel(float* out, const float* v, const float* w, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { 
        out[idx] = v[idx] - w[idx]; 
    }
}

// Kernel to add two vectors together
__global__ void addKernel(float* out, const float* v, const float* w, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { 
        out[idx] = v[idx] + w[idx]; 
    }
}

// Kernel to scale a matrix in-place
__global__
void scaleMatrixKernel(float* data, float s, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx < total) {
        data[idx] *= s;
    }
}

// Kernel to scale a vector in-place
__global__
void scaleVectorKernel(float* data, float s, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= s;
    }
}

// Kernel for left diagonal matrix multiplication: out(i,j) = d[i] * A(i,j)
__global__
void leftDiagMatMulKernel(const float* d_vec, const float* A, float* out, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        out[col * rows + row] = d_vec[row] * A[col * rows + row];
    }
}

// Kernel for right diagonal matrix multiplication: out(i,j) = A(i,j) * d[j]
__global__
void rightDiagMatMulKernel(const float* A, const float* d_vec, float* out, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        out[col * rows + row] = d_vec[col] * A[col * rows + row];
    }
}

// Kernel to compute max absolute value per block for infinity norm reduction.
__global__
void reduceAbsMaxKernel(const float* in, float* blockMax, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;
    float myMax = 0.0f;
    if (i < n) {
        myMax = fabsf(in[i]);
    }
    sdata[tid] = myMax;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s && i + s < n) {
            float other = sdata[tid + s];
            if (other > sdata[tid]) sdata[tid] = other;
        }
        __syncthreads();
    }
    // Write result for this block to global memory
    if (tid == 0) {
        blockMax[blockIdx.x] = sdata[0];
    }
}

// Custom kernel for computing A = A + alpha * I (computation performed in place)
// n is the number of rows of A.
__global__
void addScalarMatrixKernel(float* A, float alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        A[n * i + i] += alpha;
    }
}

// CUDA kernel to initialize a scaled diagonal matrix
// result = s * D, where D = diag(d)
__global__ void diagonal_kernel(float* d_matrix, const float* d_d, float s, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        d_matrix[row + col * n] = (row == col) ? d_d[row] * s : 0.0f;
    }
}

__global__ void diagonal_kernel(float* d_matrix, float s, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        d_matrix[row + col * n] = (row == col) ? s : 0.0f;
    }
}

__global__ void clipVectorKernel(float* v, const float* l, const float* u, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v_i = v[i];
        float l_i = l[i];
        float u_i = u[i];
        if (v_i < l_i) {v[i] = l_i;}
        else {
            if (v_i > u_i) {v[i] = u_i;}
        }
    }
}

// -----------------------------
// Member function implementations
// -----------------------------

void Matrix::transpose(cublasHandle_t handle, Matrix &dst) {
    if (dst.rows != cols || dst.cols != rows) {
        std::cerr << "Function transpose. Destination matrix dimensions don't match." << std::endl;
        exit(EXIT_FAILURE);
    }

    const float alpha = 1.0f, beta = 0.0f;
    cublasSgeam(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,  // Transpose A, do not transpose B
        cols, rows,                // Dimensions of transposed matrix
        &alpha,
        d_data, rows,  // Input matrix (column-major)
        &beta,
        nullptr, cols,  // No second matrix (beta = 0)
        dst.d_data, cols  // Output matrix (column-major)
    );
}

void Matrix::scale(float s) {
    int total = rows * cols;
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;
    scaleMatrixKernel<<<gridSize, blockSize>>>(d_data, s, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

void Matrix::addScalarMatrix(float alpha) {
    int blockSize = 256;
    int gridSize = (rows + blockSize - 1) / blockSize;
    addScalarMatrixKernel<<<gridSize, blockSize>>>(d_data, alpha, rows);
    CUDA_CHECK(cudaGetLastError());
}

void Vector::scale(float s) {
    // Copy original data to result
    int blockSize = 256;
    int gridSize = (n_elements + blockSize - 1) / blockSize;
    scaleVectorKernel<<<gridSize, blockSize>>>(d_data, s, n_elements);
    CUDA_CHECK(cudaGetLastError());
}

// -----------------------------
// Free Functions for Linear Algebra
// -----------------------------

// Initialization of a matrix to the identity, using custom kernels.
void eye(Matrix &dst) {
    if (dst.rows != dst.cols) {
        std::cerr << "Function eye. Matrix is not square." << std::endl;
        exit(EXIT_FAILURE);
    }
    int n = dst.rows;
    // Define CUDA grid/block dimensions
    dim3 blockDim(16, 16);  // 16x16 threads per block
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);
    // Launch kernel
    diagonal_kernel<<<gridDim, blockDim>>>(dst.d_data, 1, n);
    // Check for errors
    CUDA_CHECK(cudaGetLastError());
}

// Vector sum using custom kernels
void vecsum(Vector &dst, const Vector &v, const Vector &w) {
    if (v.n_elements != w.n_elements || dst.n_elements != v.n_elements) {
        std::cerr << "Function vecsum. Vectors dimensions mismatch." << std::endl;
        exit(EXIT_FAILURE);
    }
    int blockSize = 256;
    int gridSize = (v.n_elements + blockSize - 1) / blockSize;
    addKernel<<<gridSize, blockSize>>>(dst.d_data, v.d_data, w.d_data, v.n_elements);
    CUDA_CHECK(cudaGetLastError());
}

// Vector difference using custom kernels: result = v - w
void vecdiff(Vector &dst, const Vector &v, const Vector &w) {
    if (v.n_elements != w.n_elements || dst.n_elements != v.n_elements) {
        std::cerr << "Function vecdiff. Vectors dimensions mismatch." << std::endl;
        exit(EXIT_FAILURE);
    }
    Vector result(v.n_elements);
    int blockSize = 256;
    int gridSize = (dst.n_elements + blockSize - 1) / blockSize;
    subKernel<<<gridSize, blockSize>>>(dst.d_data, v.d_data, w.d_data, dst.n_elements);
    CUDA_CHECK(cudaGetLastError());
}

// Matrix multiplication using cuBLAS: C = A * B
void matmul(cublasHandle_t handle, Matrix &dst, const Matrix &A, const Matrix &B) {
    if (A.cols != B.rows) {
        std::cerr << "Function matmul. Matrix dimensions mismatch for multiplication." << std::endl;
        exit(EXIT_FAILURE);
    }
    if (A.rows != dst.rows || B.cols != dst.cols) {
        std::cerr << "Function matmul. Destination matrix is not of right dimensions." << std::endl;
        exit(EXIT_FAILURE);
    }
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             A.rows, B.cols, B.rows,
                             &alpha,
                             A.d_data, A.rows,
                             B.d_data, B.rows,
                             &beta,
                             dst.d_data, dst.rows));
}

// Clipping operations applied to a vector, through custom kernels
void clip(float* v, const Vector &l, const Vector &u) {
    if (l.n_elements != u.n_elements) {
        std::cerr << "Function clip(v, l, u).\nVectors dimensions mismatch." << std::endl;
        exit(EXIT_FAILURE);
    }
    int blockSize = 256;
    int gridSize = (l.n_elements + blockSize - 1) / blockSize;
    clipVectorKernel<<<gridSize, blockSize>>>(v, l.d_data, u.d_data, l.n_elements);
    CUDA_CHECK(cudaGetLastError());
}

void diag(Matrix &dst, const Vector &d, float s) {
    if (dst.rows != d.n_elements || dst.cols != d.n_elements) {
        std::cerr << "Function diag. Matrix is not n x n." << std::endl;
        exit(EXIT_FAILURE);
    }
    int n = d.n_elements;

    // Define CUDA grid/block dimensions
    dim3 blockDim(16, 16);  // 16x16 threads per block
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);

    // Launch kernel
    diagonal_kernel<<<gridDim, blockDim>>>(dst.d_data, d.d_data, s, n);

    // Check for errors
    CUDA_CHECK(cudaGetLastError());
}

void matmul_scale(cublasHandle_t handle, Matrix &dst, const Matrix &A, const Matrix &B, float s) {
    if (A.cols != B.rows) {
        std::cerr << "Matrix dimensions mismatch for multiplication." << std::endl;
        exit(EXIT_FAILURE);
    }
    if (A.rows != dst.rows || B.cols != dst.cols) {
        std::cerr << "Function matmul_scale. Destination matrix is not of right dimensions." << std::endl;
        exit(EXIT_FAILURE);
    }
    const float alpha = s, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             A.rows, B.cols, B.rows,
                             &alpha,
                             A.d_data, A.rows,
                             B.d_data, B.rows,
                             &beta,
                             dst.d_data, dst.rows));
}

void matmul_scale_add(cublasHandle_t handle, Matrix &dst, const Matrix &A, const Matrix &B, float s, const Matrix &C) {
    if (A.cols != B.rows or C.rows != A.rows or C.cols != B.cols) {
        std::cerr << "Matrix dimensions mismatch for multiplication and addition." << std::endl;
        exit(EXIT_FAILURE);
    }
    if (A.rows != dst.rows || B.cols != dst.cols) {
        std::cerr << "Function matmul_scale_add. Destination matrix is not of right dimensions." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Allocate a new matrix
    int m = C.rows;
    int n = C.cols;
    CUDA_CHECK(cudaMemcpy(dst.d_data, C.d_data, m * n * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Perform the computation D = D + s * AB
    const float alpha = s, beta = 1;
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             A.rows, B.cols, B.rows,
                             &alpha,
                             A.d_data, A.rows,
                             B.d_data, B.rows,
                             &beta,
                             dst.d_data, dst.rows));
}

void matvecmul(cublasHandle_t handle, Vector &dst, const Matrix &W, const Vector &y) {
    if (W.cols != y.n_elements || W.rows != dst.n_elements) {
        std::cerr << "Function matvecmul. Dimension mismatch in matrix vector product." << std::endl;
        exit(EXIT_FAILURE);
    }
    const float alpha = 1.0f, beta = 0.0f;
    // Using cuBLAS for matrix-vector multiplication: result = W * y.
    CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_N,
                             W.rows, W.cols,
                             &alpha,
                             W.d_data, W.rows,
                             y.d_data, 1,
                             &beta,
                             dst.d_data, 1));
}

void matvecmul(cublasHandle_t handle, float* dst, const Matrix &W, const Vector &y) {
    if (W.cols != y.n_elements) {
        std::cerr << "Function matvecmul. Dimension mismatch in matrix vector product." << std::endl;
        exit(EXIT_FAILURE);
    }
    const float alpha = 1.0f, beta = 0.0f;
    // Using cuBLAS for matrix-vector multiplication: result = W * y.
    CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_N,
                             W.rows, W.cols,
                             &alpha,
                             W.d_data, W.rows,
                             y.d_data, 1,
                             &beta,
                             dst, 1));
}

// Affine transformation: computes Wy + b.
void affine_transformation(cublasHandle_t handle, Vector &dst, const Matrix &W, const Vector &y, const Vector &b) {
    if (W.cols != y.n_elements || W.rows != b.n_elements || W.rows != dst.n_elements) {
        std::cerr << "Function affine_transformation. Dimension mismatch in affine_transformation." << std::endl;
        exit(EXIT_FAILURE);
    }
    const float alpha = 1.0f, beta = 0.0f;
    //Vector intermediate(dst.n_elements);
    // Using cuBLAS for matrix-vector multiplication: y = W * y.
    CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_N,
                             W.rows, W.cols,
                             &alpha,
                             W.d_data, W.rows,
                             y.d_data, 1,
                             &beta,
                             dst.d_data, 1));
    // Add bias b: y = y + b
    int blockSize = 256;
    int gridSize = (y.n_elements + blockSize - 1) / blockSize;
    addKernel<<<gridSize, blockSize>>>(dst.d_data, dst.d_data, b.d_data, dst.n_elements);
    CUDA_CHECK(cudaGetLastError());
}

// Left diagonal matrix multiplication: each row i of A is scaled by d[i].
void left_diag_matmul(Matrix &dst, const Vector &d, const Matrix &A) {
    if (d.n_elements != A.rows || dst.rows != A.rows || dst.cols != A.cols) {
        std::cerr << "Function left_diag_matmul. Dimension mismatch in left_diag_matmul." << std::endl;
        exit(EXIT_FAILURE);
    }
    dim3 blockDim(16, 16);
    dim3 gridDim((A.cols + blockDim.x - 1) / blockDim.x, (A.rows + blockDim.y - 1) / blockDim.y);
    leftDiagMatMulKernel<<<gridDim, blockDim>>>(d.d_data, A.d_data, dst.d_data, A.rows, A.cols);
    CUDA_CHECK(cudaGetLastError());
}

void right_diag_matmul(Matrix &dst, const Matrix &A, const Vector &d) {
    if (d.n_elements != A.cols || dst.rows != A.rows || dst.cols != A.cols) {
        std::cerr << "Function right_diag_matmul. Dimension mismatch in right_diag_matmul." << std::endl;
        exit(EXIT_FAILURE);
    }
    dim3 blockDim(16, 16);
    dim3 gridDim((A.cols + blockDim.x - 1) / blockDim.x, (A.rows + blockDim.y - 1) / blockDim.y);
    rightDiagMatMulKernel<<<gridDim, blockDim>>>(A.d_data, d.d_data, dst.d_data, A.rows, A.cols);
    CUDA_CHECK(cudaGetLastError());
}

// Compute the infinity norm (maximum absolute value) of a vector using a reduction kernel.
float infinity_norm(const Vector &v) {
    int n = v.n_elements;
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    // Allocate temporary array for block maximums.
    float* d_blockMax;
    CUDA_CHECK(cudaMalloc(&d_blockMax, gridSize * sizeof(float)));
    size_t sharedMemSize = blockSize * sizeof(float);
    reduceAbsMaxKernel<<<gridSize, blockSize, sharedMemSize>>>(v.d_data, d_blockMax, n);
    CUDA_CHECK(cudaGetLastError());

    // Copy block results to host and reduce on CPU.
    float* h_blockMax = new float[gridSize];
    CUDA_CHECK(cudaMemcpy(h_blockMax, d_blockMax, gridSize * sizeof(float), cudaMemcpyDeviceToHost));
    float maxVal = 0.0f;
    for (int i = 0; i < gridSize; ++i) {
        if (h_blockMax[i] > maxVal)
            maxVal = h_blockMax[i];
    }
    delete[] h_blockMax;
    cudaFree(d_blockMax);
    return maxVal;
}

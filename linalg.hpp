#ifndef LINALG_HPP
#define LINALG_HPP

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

// Define the type representing real numbers based on a compile time flag

typedef float real;
#define cublasgeam cublasSgeam
#define cublasgemm cublasSgemm
#define cublasgemv cublasSgemv

/*
typedef double real;
#define cublasgeam cublasDgeam
#define cublasgemm cublasDgemm
#define cublasgemv cublasDgemv
*/


// Error checking macros
#define CUDA_CHECK(err) { \
    if(err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

#define CUBLAS_CHECK(err) { \
    if(err != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error: " << err \
                  << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

class Vector {
public:
    int n_elements;
    real* d_data;
    
    Vector();
    Vector(int n);
    ~Vector();
    void copyFromHost(const real* h_data);
    void copyFromDevice(const real* d_data);
    void copyToHost(real* h_data) const;
    void scale(real s);
};

class Matrix {
public:
    int rows, cols;
    real* d_data;

    Matrix();
    Matrix(int r, int c);
    ~Matrix();
    void copyFromHost(const real* h_data);
    void copyFromDevice(const real* d_data);
    void copyToHost(real* h_data) const;
    void transpose(cublasHandle_t handle, Matrix &dst);
    void scale(real s);
    // Adds alpha to each element on the diagonal
    void addScalarMatrix(real alpha);
};

void vecsum(Vector &dst, const Vector &v, const Vector &w);
void genvecsum(Vector &dst, const Vector &v, real a, const Vector &w, real b);
void eye(Matrix &dst);
void vecdiff(Vector &dst, const Vector &v, const Vector &w);
void elementwise_product(Vector &dst, const Vector &v, const Vector &w);
void clip(real* v, const Vector &l, const Vector &u);
void diag(Matrix &dst, const Vector &d, real s);
void matmul(cublasHandle_t handle, Matrix &dst, const Matrix &A, const Matrix &B);
void matmul_scale(cublasHandle_t handle, Matrix &dst, const Matrix &A, const Matrix &B, real s);
void matmul_scale_add(cublasHandle_t handle, Matrix &dst, const Matrix &A, const Matrix &B, real s, const Matrix &C);
void matvecmul(cublasHandle_t handle, Vector &dst, const Matrix &W, const Vector &y);
void matvecmul(cublasHandle_t handle, real* dst, const Matrix &W, const Vector &y);
void affine_transformation(cublasHandle_t handle, Vector &dst, const Matrix &W, const Vector &y, const Vector &b);
void left_diag_matmul(Matrix &dst, const Vector &d, const Matrix &A);
void right_diag_matmul(Matrix &dst, const Matrix &A, const Vector &d);
real infinity_norm(const Vector &v);

#endif
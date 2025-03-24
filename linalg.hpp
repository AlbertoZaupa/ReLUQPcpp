#ifndef CUBLAS_MANAGER_HPP
#define CUBLAS_MANAGER_HPP

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>


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
    float* d_data;
    
    Vector();
    Vector(int n);
    ~Vector();
    void copyFromHost(const float* h_data);
    void copyFromDevice(const float* d_data);
    void copyToHost(float* h_data) const;
    void scale(float s);
};

class Matrix {
public:
    int rows, cols;
    float* d_data;

    Matrix();
    Matrix(int r, int c);
    ~Matrix();
    void copyFromHost(const float* h_data);
    void copyFromDevice(const float* d_data);
    void copyToHost(float* h_data) const;
    void transpose(cublasHandle_t handle, Matrix &dst);
    void scale(float s);
    // Adds alpha to each element on the diagonal
    void addScalarMatrix(float alpha);
};

void vecsum(Vector &dst, const Vector &v, const Vector &w);
void eye(Matrix &dst);
void vecdiff(Vector &dst, const Vector &v, const Vector &w);
void clip(float* v, const Vector &l, const Vector &u);
void diag(Matrix &dst, const Vector &d, float s);
void matmul(cublasHandle_t handle, Matrix &dst, const Matrix &A, const Matrix &B);
void matmul_scale(cublasHandle_t handle, Matrix &dst, const Matrix &A, const Matrix &B, float s);
void matmul_scale_add(cublasHandle_t handle, Matrix &dst, const Matrix &A, const Matrix &B, float s, const Matrix &C);
void matvecmul(cublasHandle_t handle, Vector &dst, const Matrix &W, const Vector &y);
void matvecmul(cublasHandle_t handle, float* dst, const Matrix &W, const Vector &y);
void affine_transformation(cublasHandle_t handle, Vector &dst, const Matrix &W, const Vector &y, const Vector &b);
void left_diag_matmul(Matrix &dst, const Vector &d, const Matrix &A);
void right_diag_matmul(Matrix &dst, const Matrix &A, const Vector &d);
float infinity_norm(const Vector &v);

#endif
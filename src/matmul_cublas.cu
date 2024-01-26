#include <cublas_v2.h>
#include <cuda_runtime.h>

extern "C"
void matmul_cublas(float *a, float *b, float *c, int m, int n, int k) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.;
    float beta = 0.;

    // Note: swapped m and k, and a and b, also using CUBLAS_OP_T for transpose
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, n, m, k, &alpha, b, k, a, m, &beta, c, n);

    cublasDestroy(handle);
}
#include <cublas_v2.h>
#include <cuda_runtime.h>

extern "C"
void batchedMatMulKernel(float **Aarray, float **Barray, float **Carray, int m, int k, int n, int batchCount) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Perform batched matrix multiplication
    cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                       m, n, k,
                       &alpha,
                       (const float **)Aarray, m,
                       (const float **)Barray, k,
                       &beta,
                       Carray, m,
                       batchCount);

    cublasDestroy(handle);
}


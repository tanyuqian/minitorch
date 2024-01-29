#include <cuda_runtime.h>

#include <iostream>
#include <sstream>
#include <fstream>

#define BLOCK_DIM 1024
#define MAX_DIMS 10
#define TILE 32
// typedef float float;

#define ADD_FUNC       1
#define MUL_FUNC       2
#define ID_FUNC        3
#define NEG_FUNC       4
#define LT_FUNC        5
#define EQ_FUNC        6
#define SIGMOID_FUNC   7
#define RELU_FUNC      8
#define RELU_BACK_FUNC 9
#define LOG_FUNC       10
#define LOG_BACK_FUNC  11
#define EXP_FUNC       12
#define INV_FUNC       13
#define INV_BACK_FUNC  14
#define IS_CLOSE_FUNC  15
#define MAX_FUNC       16
#define POW            17
#define TANH           18

__device__ float fn(int fn_id, float x, float y=0) {
    switch(fn_id) {
      case ADD_FUNC: {
        return x + y;
      }
      case MUL_FUNC: {
        return x * y;
      }
      case ID_FUNC: {
      	return x;
      }
      case NEG_FUNC: {
        return -x;
      }
      case LT_FUNC: {
        if (x < y) {
          return 1.0;
        }
        else {
          return 0.0;
        }
      }
      case EQ_FUNC: {
        if (x == y) {
          return 1.0;
        }
        else {
          return 0.0;
        }
      }
      case SIGMOID_FUNC: {
        if (x >= 0) {
          return 1.0 / (1.0 + exp(-x));
        }
        else {
          return exp(x) / (1.0 + exp(x));
        }
      }
      case RELU_FUNC: {
        return max(x, 0.0);
      }
      case RELU_BACK_FUNC: {
        if (x > 0) {
          return y;
        }
        else {
          return 0.0;
        }
      }
      case LOG_FUNC: {
        return log(x + 1e-6);
      }
      case LOG_BACK_FUNC: {
        return y / (x + 1e-6);
      }
      case EXP_FUNC: {
        return exp(x);
      }
      case INV_FUNC: {
        return float(1.0 / x);
      }
      case INV_BACK_FUNC: {
        return -(1.0 / (x * x)) * y;
      }
      case IS_CLOSE_FUNC: {
        return (x - y < 1e-2) && (y - x < 1e-2);
      }
      case MAX_FUNC: {
        if (x > y) {
          return x;
        }
        else {
          return y;
        }
      }
      case POW: {
        return pow(x, y);
      }
      case TANH: {
        return tanh(x);
      }
      default: {
        return x + y;
      }
    }
    
}


__device__ int index_to_position(const int* index, const int* strides, int num_dims) {
    int position = 0;
    for (int i = 0; i < num_dims; ++i) {
        position += index[i] * strides[i];
    }
    return position;
}

__device__ void to_index(int ordinal, const int* shape, int* out_index, int num_dims) {
    int cur_ord = ordinal;
    for (int i = num_dims - 1; i >= 0; --i) {
        int sh = shape[i];
        out_index[i] = cur_ord % sh;
        cur_ord /= sh;
    }
}

__device__ void broadcast_index(const int* big_index, const int* big_shape, const int* shape, int* out_index, int num_dims_big, int num_dims) {
    for (int i = 0; i < num_dims; ++i) {
        if (shape[i] > 1) {
            out_index[i] = big_index[i + (num_dims_big - num_dims)];
        } else {
            out_index[i] = 0;
        }
    }
}


__global__ void MatrixMultiplyKernel(
    float* out,
    const int* out_shape,
    const int* out_strides,
    float* a_storage,
    const int* a_shape,
    const int* a_strides,
    float* b_storage,
    const int* b_shape,
    const int* b_strides
) {

    __shared__ float a_shared[32][32];
    __shared__ float b_shared[32][32];

    int batch = blockIdx.z;
    int a_batch_stride = a_shape[0] > 1 ? a_strides[0] : 0;
    int b_batch_stride = b_shape[0] > 1 ? b_strides[0] : 0;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int pi = threadIdx.x;
    int pj = threadIdx.y;

    float accum = 0.0;
    // printf("Start working: %d, %d, %d, %d", i, j, pi, pj);
    for (int k_start = 0; k_start < a_shape[2]; k_start += TILE) {
        int k = k_start + pj;
        if (i < a_shape[1] && k < a_shape[2]) {
            a_shared[pi][pj] = a_storage[a_batch_stride * batch + a_strides[1] * i + a_strides[2] * k];
        } else {
            a_shared[pi][pj] = 0.0;
        }

        k = k_start + pi;
        if (j < b_shape[2] && k < b_shape[1]) {
            b_shared[pi][pj] = b_storage[b_batch_stride * batch + b_strides[1] * k + b_strides[2] * j];
        } else {
            b_shared[pi][pj] = 0.0;
        }

        __syncthreads();

        for (k = 0; k < TILE; ++k) {
            if ((k_start + k) < a_shape[2]) {
                accum += a_shared[pi][k] * b_shared[k][pj];
            }
        }

        __syncthreads();
    }

    if (i < out_shape[1] && j < out_shape[2]) {
        out[out_strides[0] * batch + out_strides[1] * i + out_strides[2] * j] = accum;
    }
}


__global__ void mapKernel(
    float* out, 
    int* out_shape, 
    int* out_strides, 
    int out_size, 
    float* in_storage, 
    int* in_shape, 
    int* in_strides,
    int shape_size,
    int fn_id
) {
    int out_index[MAX_DIMS];
    int in_index[MAX_DIMS];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < out_size) {
        to_index(i, out_shape, out_index, shape_size);
        broadcast_index(out_index, out_shape, in_shape, in_index, shape_size, shape_size);
        int o = index_to_position(out_index, out_strides, shape_size);
        int j = index_to_position(in_index, in_strides, shape_size);
        // printf("out[%d] = fn(in[%d]) = %f\n", o, j, fn(fn_id, in_storage[j]));
        out[o] = fn(fn_id, in_storage[j]);
    }
}


__global__ void reduceKernel(
    float* out, 
    int* out_shape, 
    int* out_strides, 
    int out_size, 
    float* a_storage, 
    int* a_shape, 
    int* a_strides, 
    int reduce_dim,
    float reduce_value,
    int shape_size,
    int fn_id
) {
    int out_index[MAX_DIMS];
    int out_pos = blockIdx.x * blockDim.x + threadIdx.x;;
    if (out_pos < out_size) {
      out[out_pos] = reduce_value;
      to_index(out_pos, out_shape, out_index, shape_size);
      int o_pos = index_to_position(out_index, out_strides, shape_size);
      // printf("out[%d, %d] = [%f]\n", out_pos, o_pos, out[out_pos]);
      // printf("reduce_dim: %d\n", a_shape[reduce_dim]);
      for(int i = 0; i < a_shape[reduce_dim]; i++) {
        out_index[reduce_dim] = i;
        int i_pos = index_to_position(out_index, a_strides, shape_size);
        out[out_pos] = fn(fn_id, out[out_pos], a_storage[i_pos]);
        // printf("out[%d] = [%f], a_storage[%d] = [%f]\n", out_pos, out[out_pos], i_pos, a_storage[i_pos]);
      }
    }
}

__global__ void zipKernel(
    float* out, 
    int* out_shape, 
    int* out_strides, 
    int out_size,
    int out_shape_size,
    float* a_storage, 
    int* a_shape, 
    int* a_strides,
    int a_shape_size,
    float* b_storage, 
    int* b_shape, 
    int* b_strides,
    int b_shape_size,
    int fn_id
) {
    int out_index[MAX_DIMS];
    int a_index[MAX_DIMS];
    int b_index[MAX_DIMS];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < out_size) {
        to_index(i, out_shape, out_index, out_shape_size);
        int o = index_to_position(out_index, out_strides, out_shape_size);
        broadcast_index(out_index, out_shape, a_shape, a_index, out_shape_size, a_shape_size);
        int j = index_to_position(a_index, a_strides, a_shape_size);
        broadcast_index(out_index, out_shape, b_shape, b_index, out_shape_size, b_shape_size);
        int k = index_to_position(b_index, b_strides, b_shape_size);
        out[o] = fn(fn_id, a_storage[j], b_storage[k]);
    }
}


extern "C" {

void MatrixMultiply(
    float* out,
    int* out_shape,
    int* out_strides,
    float* a_storage,
    int* a_shape,
    int* a_strides,
    float* b_storage,
    int* b_shape,
    int* b_strides,
    int batch, int m, int p
) {
    int n = a_shape[2];

    // Allocate device memory
    float *d_out, *d_a, *d_b;
    cudaError_t status = cudaMalloc(&d_a, batch * m * n * sizeof(float));
    if (status != cudaSuccess) {
      fprintf(stderr, "Matmul Malloc Matrix A Error: %s\n", cudaGetErrorString(status));
      exit(EXIT_FAILURE);
    }
    status = cudaMalloc(&d_b, batch * n * p * sizeof(float));
    if (status != cudaSuccess) {
      fprintf(stderr, "Matmul Malloc Matrix B Error: %s\n", cudaGetErrorString(status));
      exit(EXIT_FAILURE);
    }
    status = cudaMalloc(&d_out, batch * m * p * sizeof(float));
    if (status != cudaSuccess) {
      fprintf(stderr, "Matmul Malloc Matrix OUT Error: %s\n", cudaGetErrorString(status));
      exit(EXIT_FAILURE);
    }

    int *d_out_shape, *d_out_strides, *d_a_shape, *d_a_strides, *d_b_shape, *d_b_strides;
    status = cudaMalloc(&d_out_shape, 3 * sizeof(int));
    if (status != cudaSuccess) {
      fprintf(stderr, "Matmul Malloc Matrix out_shape Error: %s\n", cudaGetErrorString(status));
      exit(EXIT_FAILURE);
    }
    status = cudaMalloc(&d_out_strides, 3 * sizeof(int));
    if (status != cudaSuccess) {
      fprintf(stderr, "Matmul Malloc Matrix out_strides Error: %s\n", cudaGetErrorString(status));
      exit(EXIT_FAILURE);
    }
    status = cudaMalloc(&d_a_shape, 3 * sizeof(int));
    if (status != cudaSuccess) {
      fprintf(stderr, "Matmul Malloc Matrix a_shape Error: %s\n", cudaGetErrorString(status));
      exit(EXIT_FAILURE);
    }
    status = cudaMalloc(&d_a_strides, 3 * sizeof(int));
    if (status != cudaSuccess) {
      fprintf(stderr, "Matmul Malloc Matrix a_strides Error: %s\n", cudaGetErrorString(status));
      exit(EXIT_FAILURE);
    }
    status = cudaMalloc(&d_b_shape, 3 * sizeof(int));
    if (status != cudaSuccess) {
      fprintf(stderr, "Matmul Malloc Matrix b_shape Error: %s\n", cudaGetErrorString(status));
      exit(EXIT_FAILURE);
    }
    status = cudaMalloc(&d_b_strides, 3 * sizeof(int));
    if (status != cudaSuccess) {
      fprintf(stderr, "Matmul Malloc Matrix b_strides Error: %s\n", cudaGetErrorString(status));
      exit(EXIT_FAILURE);
    }

    // Copy data to the device
    cudaMemcpy(d_a, a_storage, batch * m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_storage, batch * n * p * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_shape, out_shape, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_strides, out_strides, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_shape, a_shape, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_strides, a_strides, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_shape, b_shape, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_strides, b_strides, 3 * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 32;
    dim3 blockDims(threadsPerBlock, threadsPerBlock, 1); // Adjust these values based on your specific requirements
    dim3 gridDims((m + threadsPerBlock - 1) / threadsPerBlock, (p + threadsPerBlock - 1) / threadsPerBlock, batch);
    MatrixMultiplyKernel<<<gridDims, blockDims>>>(
        d_out, d_out_shape, d_out_strides, d_a, d_a_shape, d_a_strides, d_b, d_b_shape, d_b_strides
    );

    // Copy back to the host
    cudaMemcpy(out, d_out, batch * m * p * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();

    // Check CUDA execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "Matmul Error: %s\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    cudaFree(d_out_shape);
    cudaFree(d_out_strides);
    cudaFree(d_a_shape);
    cudaFree(d_a_strides);
    cudaFree(d_b_shape);
    cudaFree(d_b_strides);
}

void tensorMap(
    float* out, 
    int* out_shape, 
    int* out_strides, 
    int out_size, 
    float* in_storage, 
    int* in_shape, 
    int* in_strides,
    int in_size,
    int shape_size,
    int fn_id
) {

    float *d_out, *d_in;
    cudaError_t status = cudaMalloc(&d_out, out_size * sizeof(float));
    if (status != cudaSuccess) {
      fprintf(stderr, "Map Malloc Matrix out Error: %s\n", cudaGetErrorString(status));
      exit(EXIT_FAILURE);
    }
    status = cudaMalloc(&d_in, in_size * sizeof(float));
    if (status != cudaSuccess) {
      fprintf(stderr, "Map Malloc Matrix in Error: %s\n", cudaGetErrorString(status));
      exit(EXIT_FAILURE);
    }

    int *d_out_shape, *d_out_strides, *d_in_shape, *d_in_strides;
    status = cudaMalloc(&d_out_shape, shape_size * sizeof(int));
    if (status != cudaSuccess) {
      fprintf(stderr, "Map Malloc out_shape Error: %s\n", cudaGetErrorString(status));
      exit(EXIT_FAILURE);
    }
    status = cudaMalloc(&d_out_strides, shape_size * sizeof(int));
    if (status != cudaSuccess) {
      fprintf(stderr, "Map Malloc out_strides Error: %s\n", cudaGetErrorString(status));
      exit(EXIT_FAILURE);
    }
    status = cudaMalloc(&d_in_shape, shape_size * sizeof(int));
    if (status != cudaSuccess) {
      fprintf(stderr, "Map Malloc in_shape Error: %s\n", cudaGetErrorString(status));
      exit(EXIT_FAILURE);
    }
    status = cudaMalloc(&d_in_strides, shape_size * sizeof(int));
    if (status != cudaSuccess) {
      fprintf(stderr, "Map Malloc in_strides Error: %s\n", cudaGetErrorString(status));
      exit(EXIT_FAILURE);
    }

    cudaMemcpy(d_in, in_storage, in_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_shape, out_shape, shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_strides, out_strides, shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_shape, in_shape, shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_strides, in_strides, shape_size * sizeof(int), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 32;
    int blocksPerGrid = (out_size + threadsPerBlock - 1) / threadsPerBlock;
    mapKernel<<<blocksPerGrid, threadsPerBlock>>>(
      d_out, d_out_shape, d_out_strides, out_size, 
      d_in, d_in_shape, d_in_strides, 
      shape_size, fn_id);
    
    // Copy back to the host
    cudaMemcpy(out, d_out, out_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Check CUDA execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "Map Error: %s\n", cudaGetErrorString(err));
      // Handle the error (e.g., by exiting the program)
      exit(EXIT_FAILURE);
    }

    // Free memory on device
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_out_shape);
    cudaFree(d_out_strides);
    cudaFree(d_in_shape);
    cudaFree(d_in_strides);
}


void tensorZip(
    float* out, 
    int* out_shape, 
    int* out_strides, 
    int out_size,
    int out_shape_size,
    float* a_storage, 
    int* a_shape, 
    int* a_strides,
    int a_size,
    int a_shape_size,
    float* b_storage, 
    int* b_shape, 
    int* b_strides,
    int b_size,
    int b_shape_size,
    int fn_id
) {

    // Allocate device memory
    float *d_out, *d_a, *d_b;
    cudaError_t status = cudaMalloc(&d_a, a_size * sizeof(float));
    status = cudaMalloc(&d_b, b_size * sizeof(float));
    status = cudaMalloc(&d_out, out_size * sizeof(float));

    int *d_out_shape, *d_out_strides, *d_a_shape, *d_a_strides, *d_b_shape, *d_b_strides;
    status = cudaMalloc(&d_out_shape, out_shape_size * sizeof(int));
    status = cudaMalloc(&d_out_strides, out_shape_size * sizeof(int));
    status = cudaMalloc(&d_a_shape, a_shape_size * sizeof(int));
    status = cudaMalloc(&d_a_strides, a_shape_size * sizeof(int));
    status = cudaMalloc(&d_b_shape, b_shape_size * sizeof(int));
    status = cudaMalloc(&d_b_strides, b_shape_size * sizeof(int));

    // Copy data to the device
    cudaMemcpy(d_a, a_storage, a_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_storage, b_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_shape, out_shape, out_shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_strides, out_strides, out_shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_shape, a_shape, a_shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_strides, a_strides, a_shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_shape, b_shape, b_shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_strides, b_strides, b_shape_size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 32;
    int blocksPerGrid = (out_size + threadsPerBlock - 1) / threadsPerBlock;
    zipKernel<<<blocksPerGrid, threadsPerBlock>>>(
      d_out, d_out_shape, d_out_strides, out_size, out_shape_size,
      d_a, d_a_shape, d_a_strides, a_shape_size,
      d_b, d_b_shape, d_b_strides, b_shape_size,
      fn_id);

    cudaMemcpy(out, d_out, out_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();


    // Check CUDA execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "Zip Error: %s\n", cudaGetErrorString(err));
      // Handle the error (e.g., by exiting the program)
      exit(EXIT_FAILURE);
    }

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    cudaFree(d_out_shape);
    cudaFree(d_out_strides);
    cudaFree(d_a_shape);
    cudaFree(d_a_strides);
    cudaFree(d_b_shape);
    cudaFree(d_b_strides);
}



void tensorReduce(
    float* out, 
    int* out_shape, 
    int* out_strides, 
    int out_size, 
    float* a_storage, 
    int* a_shape, 
    int* a_strides, 
    int reduce_dim, 
    float reduce_value,
    int shape_size,
    int fn_id
) {
    int a_size = out_size * a_shape[reduce_dim];
    float *d_out, *d_a;
    cudaMalloc(&d_out, out_size * sizeof(float));
    cudaMalloc(&d_a, a_size * sizeof(float));

    int *d_out_shape, *d_out_strides, *d_a_shape, *d_a_strides;
    cudaMalloc(&d_out_shape, shape_size * sizeof(int));
    cudaMalloc(&d_out_strides, shape_size * sizeof(int));
    cudaMalloc(&d_a_shape, shape_size * sizeof(int));
    cudaMalloc(&d_a_strides, shape_size * sizeof(int));

    cudaMemcpy(d_a, a_storage, a_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_shape, out_shape, shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_strides, out_strides, shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_shape, a_shape, shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_strides, a_strides, shape_size * sizeof(int), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 32;
    int blocksPerGrid = (out_size + threadsPerBlock - 1) / threadsPerBlock;
    reduceKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_out, d_out_shape, d_out_strides, out_size, 
        d_a, d_a_shape, d_a_strides, 
        reduce_dim, reduce_value, shape_size, fn_id
    );

    cudaMemcpy(out, d_out, out_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();

    // Check CUDA execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "Reduce Error: %s\n", cudaGetErrorString(err));
      // Handle the error (e.g., by exiting the program)
      exit(EXIT_FAILURE);
    }

    cudaFree(d_a);
    cudaFree(d_out);
    cudaFree(d_out_shape);
    cudaFree(d_out_strides);
    cudaFree(d_a_shape);
    cudaFree(d_a_strides);
}

}

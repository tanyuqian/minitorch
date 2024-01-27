mkdir minitorch/cuda_kernels
nvcc -o minitorch/cuda_kernels/combine.so --shared src/combine.cu -Xcompiler -fPIC
nvcc -o minitorch/cuda_kernels/batched_matmul.so --shared src/batched_matmul_cublas.cu -lcublas
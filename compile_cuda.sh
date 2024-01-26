nvcc -o minitorch/cuda_kernels/combine.so --shared src/combine.cu -Xcompiler -fPIC
nvcc -o minitorch/cuda_kernels/matmul.so --shared src/matmul_cublas.cu -lcublas
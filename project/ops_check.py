from numba import njit
import random
import numpy as np
import torch
import minitorch
import minitorch.fast_ops
from minitorch.cuda_kernel_ops import CudaKernelOps
BACKEND = minitorch.TensorBackend(CudaKernelOps)

# MAP
def map_test():
  print("MAP: ", end="")

  def test_log():
    x = [[random.random() for i in range(2)] for j in range(2)]
    z = minitorch.tensor(
      x, backend=BACKEND
    )
    z_log = z.log()
    np.testing.assert_allclose(
      z_log._tensor._storage, 
      np.log(np.array(x)).reshape(-1),
      atol=1e-5, rtol=1e-5)
    
  test_log()

  def test_inv():
    x = [[random.random() for i in range(2)] for j in range(2)]
    x_inv = [[1.0 / x[i][j] for j in range(2)] for i in range(2)]
    z = minitorch.tensor(
      x, backend=BACKEND
    )
    z_inv = z.f.inv_map(z)
    np.testing.assert_allclose(
      z_inv._tensor._storage, 
      np.array(x_inv).reshape(-1),
      atol=1e-5, rtol=1e-5)

  test_inv()
  print("SMALL CHECK DONE!")

# ZIP
def zip_test():
  print("ZIP: ", end="")
  x1 = [[random.random() for j in range(10)] for i in range(20)]
  y1 = [[random.random() for j in range(10)] for i in range(20)]
  x = minitorch.tensor(x1, backend=BACKEND)
  y = minitorch.tensor(y1, backend=BACKEND)

  np.testing.assert_allclose(
      (x + y)._tensor._storage, 
      (np.array(x1) + np.array(y1)).reshape(-1),
      atol=1e-5, rtol=1e-5)

  np.testing.assert_allclose(
      (x * y)._tensor._storage, 
      (np.array(x1) * np.array(y1)).reshape(-1),
      atol=1e-5, rtol=1e-5)
  print("SMALL CHECK DONE!")

# MM
def matmul_test():
  print("MATRIX MULTIPLY: ", end="")
  out, a, b = (
      minitorch.zeros((2, 2)),
      minitorch.zeros((2, 2)),
      minitorch.zeros((2, 2)),
  )

  x1 = [[random.random() for j in range(10)] for i in range(20)]
  y1 = [[random.random() for j in range(20)] for i in range(10)]

  z = minitorch.tensor(x1, backend=BACKEND) @ minitorch.tensor(
      y1, backend=BACKEND)
  np.testing.assert_allclose(
      z._tensor._storage, 
      (np.array(x1) @ np.array(y1)).reshape(-1),
      atol=1e-5, rtol=1e-5)
  print("SMALL CHECK DONE!")

# REDUCE
def reduce_test():
  print("REDUCE: ", end="")
  max_reduce = minitorch.CudaKernelOps.reduce(minitorch.operators.max, -1e9)
  x = [[random.random() for i in range(2)] for j in range(2)]
  z = minitorch.tensor(
    x, backend=BACKEND
  )
  mr = max_reduce(z, 1)
  np.testing.assert_allclose(
      mr._tensor._storage, 
      np.max(np.array(x), axis=1),
      atol=1e-5, rtol=1e-5)

  x = [[random.random() for i in range(2)] for j in range(2)]
  z = minitorch.tensor(
    x, backend=BACKEND
  )
  sr = z.sum(0)
  np.testing.assert_allclose(
      sr._tensor._storage, 
      np.sum(np.array(x), axis=0),
      atol=1e-5, rtol=1e-5)
  print("SMALL CHECK DONE!")


def matmul_4d_test():
  print("MATRIX MULTIPLY 4D: ", end="")

  x1 = np.random.randn(1, 2, 3, 4)
  y1 = np.random.randn(1, 2, 4, 3)
  
  x1_tensor = torch.tensor(x1)
  y1_tensor = torch.tensor(y1)
  
  x = minitorch.tensor_from_numpy(x1, backend=BACKEND) 
  y = minitorch.tensor_from_numpy(y1, backend=BACKEND)

  z = x @ y

  np.testing.assert_allclose(
      z._tensor._storage, 
      torch.matmul(x1_tensor, y1_tensor).numpy().reshape(-1),
      atol=1e-5, rtol=1e-5)
  print("SMALL CHECK DONE!")


if __name__ == "__main__":
  # map_test()
  # zip_test()
  # reduce_test()
  # matmul_test()
  matmul_4d_test()

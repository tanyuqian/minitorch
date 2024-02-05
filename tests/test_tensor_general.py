import random
from typing import Callable, Dict, Iterable, List, Tuple
import numpy as np
import numba
import pytest
import torch
from hypothesis import given, settings
from hypothesis.strategies import DataObject, data, integers, lists, permutations

import os

import minitorch
from minitorch import MathTestVariable, Tensor, TensorBackend, grad_check

from .strategies import assert_close, small_floats
from .tensor_strategies import assert_close_tensor, shaped_tensors, tensors

one_arg, two_arg, red_arg = MathTestVariable._comp_testing()

shared: Dict[str, TensorBackend] = {}
from minitorch.cuda_kernel_ops import CudaKernelOps


if numba.cuda.is_available():
    backend_tests = [pytest.param("cuda")]
    matmul_tests = [pytest.param("cuda")]
    shared["cuda"] = minitorch.TensorBackend(CudaKernelOps)


@given(lists(small_floats, min_size=1))
@pytest.mark.parametrize("backend", backend_tests)
def test_create(backend: str, t1: List[float]) -> None:
    "Create different tensors."
    t2 = minitorch.tensor(t1, backend=shared[backend])
    for i in range(len(t1)):
        # print(f"Debug in create: {type(t1[i])}, {type(t2[i])}")
        np.testing.assert_allclose(t1[i], t2[i], atol=1e-5, rtol=1e-5)
        # assert t1[i] == t2[i]


@given(data())
@settings(max_examples=100)
@pytest.mark.parametrize("fn", one_arg)
@pytest.mark.parametrize("backend", backend_tests)
def test_cuda_one_args(
    fn: Tuple[str, Callable[[float], float], Callable[[Tensor], Tensor]],
    backend: str,
    data: DataObject,
) -> None:
    "Run forward for all one arg functions above."
    t1 = data.draw(tensors(backend=shared[backend]))
    name, base_fn, tensor_fn = fn
    t2 = tensor_fn(t1)
    for ind in t2._tensor.indices():
      assert_close(t2[ind], base_fn(t1[ind]))


@given(data())
@settings(max_examples=100)
@pytest.mark.parametrize("fn", two_arg)
@pytest.mark.parametrize("backend", backend_tests)
def test_cuda_two_args(
    fn: Tuple[str, Callable[[float, float], float], Callable[[Tensor, Tensor], Tensor]],
    backend: str,
    data: DataObject,
) -> None:
    "Run forward for all two arg functions above."
    t1, t2 = data.draw(shaped_tensors(2, backend=shared[backend]))
    name, base_fn, tensor_fn = fn
    t3 = tensor_fn(t1, t2)
    for ind in t3._tensor.indices():
        assert_close(t3[ind], base_fn(t1[ind], t2[ind]))


@given(data())
@pytest.mark.parametrize("fn", one_arg)
@pytest.mark.parametrize("backend", backend_tests)
def test_cuda_one_derivative(
    fn: Tuple[str, Callable[[float], float], Callable[[Tensor], Tensor]],
    backend: str,
    data: DataObject,
) -> None:
    "Run backward for all one arg functions above."
    t1 = data.draw(tensors(backend=shared[backend]))
    name, _, tensor_fn = fn
    grad_check(tensor_fn, t1)


@given(data())
@settings(max_examples=50)
@pytest.mark.parametrize("fn", two_arg)
@pytest.mark.parametrize("backend", backend_tests)
def test_cuda_two_grad(
    fn: Tuple[str, Callable[[float, float], float], Callable[[Tensor, Tensor], Tensor]],
    backend: str,
    data: DataObject,
) -> None:
    "Run backward for all two arg functions above."
    t1, t2 = data.draw(shaped_tensors(2, backend=shared[backend]))
    name, _, tensor_fn = fn
    grad_check(tensor_fn, t1, t2)


@given(data())
@settings(max_examples=24)
@pytest.mark.parametrize("fn", two_arg)
@pytest.mark.parametrize("backend", backend_tests)
def test_cuda_two_grad_broadcast(
    fn: Tuple[str, Callable[[float, float], float], Callable[[Tensor, Tensor], Tensor]],
    backend: str,
    data: DataObject,
) -> None:
    "Run backward for all two arg functions above with broadcast."
    t1, t2 = data.draw(shaped_tensors(2, backend=shared[backend]))
    name, base_fn, tensor_fn = fn

    grad_check(tensor_fn, t1, t2)

    # broadcast check
    grad_check(tensor_fn, t1.sum(0), t2)
    grad_check(tensor_fn, t1, t2.sum(0))


@given(data())
@settings(max_examples=100, derandomize=True)
@pytest.mark.parametrize("fn", red_arg)
@pytest.mark.parametrize("backend", backend_tests)
def test_cuda_reduce(
    fn: Tuple[str, Callable[[Iterable[float]], float], Callable[[Tensor], Tensor]],
    backend: str,
    data: DataObject,
) -> None:
    "Run backward for all reduce functions above."
    t1 = data.draw(tensors(backend=shared[backend]))
    name, _, tensor_fn = fn
    grad_check(tensor_fn, t1)


@pytest.mark.parametrize("fn", red_arg)
@pytest.mark.parametrize("backend", backend_tests)
def test_cuda_reduce_single(
    fn: Tuple[str, Callable[[Iterable[float]], float], Callable[[Tensor], Tensor]],
    backend: str,
) -> None:
    "Run backward for all reduce functions above."
    t1 = minitorch.tensor([
	[99.186523, 29.000000]], backend=shared[backend], requires_grad=True)
    name, _, tensor_fn = fn
    grad_check(tensor_fn, t1)


@pytest.mark.parametrize("backend", backend_tests)
def test_cuda_reduce_sum_practice1(
    backend: str,
) -> None:
    x = [random.random() for i in range(32)]
    b = minitorch.tensor(x)
    s = b.sum()[0]
    b2 = minitorch.tensor(x, backend=shared[backend])
    out = b2.sum(0)
    assert_close(s, out[0])


@pytest.mark.parametrize("backend", backend_tests)
def test_cuda_reduce_sum_practice2(
    backend: str,
) -> None:
    x = [random.random() for i in range(500)]
    b = minitorch.tensor(x)
    s = b.sum()[0]
    b2 = minitorch.tensor(x, backend=shared[backend])
    out = b2.sum(0)
    assert_close(s, out[0])


@pytest.mark.parametrize("backend", backend_tests)
def test_cuda_reduce_sum_practice3(
    backend: str,
) -> None:
    x = [[random.random() for i in range(32)] for j in range(16)]
    b = minitorch.tensor(x)
    s = b.sum(1)
    b2 = minitorch.tensor(x, backend=shared[backend])
    out = b2.sum(1)
    for i in range(16):
        assert_close(s[i, 0], out[i, 0])

    
matmul_dims = [
    (2, 2, 2),
    (33, 33, 33),
    (16, 16, 16),
    (8, 8, 8),
    (1, 2, 3),
    (3, 4, 5),
    (5, 4, 3),
    (64, 64, 64),
    (72, 72, 72),
    (72, 73, 74),
    (74, 73, 72),
    (128, 128, 128),
]


@pytest.mark.parametrize("m,n,p", matmul_dims)
@pytest.mark.parametrize("backend", matmul_tests)
def test_cuda_matmul_numpy_eq(m, n, p, backend):
    _a = [[random.random() for j in range(n)] for i in range(m)]
    _b = [[random.random() for j in range(p)] for i in range(n)]
    c = minitorch.tensor(_a, backend=shared[backend]) @ minitorch.tensor(
        _b, backend=shared[backend])
    _c = np.array(_a) @ np.array(_b)
    np.testing.assert_allclose(
      c.to_numpy(), _c,
      atol=1e-5, rtol=1e-5)

@pytest.mark.parametrize("backend", matmul_tests)
def test_cuda_matmul_transpose(
    backend: str
) -> None:
    "non-square matrix multiplication"
    y1 = [[1.060827],[1.088836]]
    x1 = [
    [0.078633, -0.014161, -0.004430, -0.026322, 0.025413, 0.053841, -0.019814, -0.030969, 0.030927, -0.017623, 0.043520, 0.017807, 0.027121, -0.014492, -0.002200, 0.023008, -0.051772, -0.015968, -0.026916, -0.025462, 0.022397, 0.006308, 0.041307, -0.017634 ,-0.063493, -0.138394, -0.016439, -0.001476 ,0.039018, -0.028352, 0.360597, 0.021912, -0.014997 ,-0.060067, 0.015498 ,0.037198, -0.004001, -0.056578, 0.017538, -0.020992, 0.002291, -0.011406, 0.025177, 0.044035, -0.011892, -0.032769, 0.007307, -0.010899, -0.025381, -0.029157],
    [0.025094, 0.021550,-0.038259 ,-0.021197 ,0.030559, 0.059986, 0.015934, -0.028781, -0.033896, 0.048600, -0.042933, -0.001561, -0.002027, -0.024751, 0.034852, -0.006681, 0.018085, 0.005446, -0.026051, -0.047808, -0.012565, 0.063337, 0.059277, -0.013247 ,0.041941, -0.105160 ,-0.059339, 0.045373, 0.057688, -0.033190, 0.324249 ,0.006693 ,0.018170 ,-0.019590, 0.017880 ,0.051460, 0.018164 ,0.027205, -0.000471 ,0.000983 ,0.039456 ,0.049255, -0.001225, 0.038359, -0.015454, 0.025885, 0.005716, 0.017060 ,-0.009751 ,0.007455]]
    
    def transpose(a: Tensor) -> Tensor:
        order = list(range(a.dims))
        order[-2], order[-1] = order[-1], order[-2]
        return a._new(a._tensor.permute(*order))

    x = minitorch.tensor(x1, backend=shared[backend])
    y = minitorch.tensor(y1, backend=shared[backend])
    
    z = transpose(x) @ y
    
    np.testing.assert_allclose(
      z.to_numpy(), np.array(x1).T @ np.array(y1),
      atol=1e-5, rtol=1e-5)


@given(data())
@settings(max_examples=100)
@pytest.mark.parametrize("backend", backend_tests)
def test_cuda_permute(backend: str, data: DataObject) -> None:
    "Check permutations for all backends."
    t1 = data.draw(tensors(backend=shared[backend]))
    permutation = data.draw(permutations(range(len(t1.shape))))

    def permute(a: Tensor) -> Tensor:
        return a.permute(*permutation)

    minitorch.grad_check(permute, t1)


@pytest.mark.parametrize("dims", [(2,3,4,5,6), (10, 5, 15, 20, 30), (16, 16, 16, 16, 16)])
@pytest.mark.parametrize("backend", backend_tests)
def test_bmm(dims, backend):
    b, h, n, m, p = dims
    d1 = np.random.randn(b, h, n, m).tolist() # (b, h, n, m)
    d2 = np.random.randn(b, h, m, p).tolist() # (b, h, m, p)
    
    x1 = minitorch.tensor(d1, backend=shared[backend])
    x2 = minitorch.tensor(d2, backend=shared[backend])
    result = x1 @ x2
    
    x1_ = torch.tensor(d1)
    x2_ = torch.tensor(d2)
    result_ = x1_ @ x2_

    np.testing.assert_allclose(
        result.to_numpy(),
        result_.detach().numpy(),
        atol=1e-5, 
        rtol=1e-5
    )

###############################################################################
# Assignment 2 Problem 1
###############################################################################


################################ POW 1 ########################################

@pytest.mark.ref_student_a2_1
@pytest.mark.parametrize("sizes", [(5,), (128,), (1, 64), (128, 256)])
@pytest.mark.parametrize("exp", [0, 1, 2, 3])
@pytest.mark.parametrize("backend", backend_tests)
def test_ref_student_pow_1(sizes, exp, backend):
    np.random.seed(10)
    test_dir = f'./tests/data/pow_1'
    test_str = str(exp)+'_' + '_'.join(map(str, sizes))
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    data = np.random.randn(*sizes)
    x = minitorch.tensor(data.tolist(), backend=shared[backend], requires_grad=True)
    x_ = torch.tensor(data, dtype=torch.float32, requires_grad=True)

    result = x ** exp
    result_ = x_ ** exp

    np.testing.assert_allclose(
        result.to_numpy(),
        result_.detach().numpy(),
        atol=1e-5, 
        rtol=1e-5
    )

    # Backward
    result.sum().backward()
    result_.sum().backward()

    np.testing.assert_allclose(
        x.grad.to_numpy(), 
        x_.grad.detach().numpy(), 
        atol=1e-5, 
        rtol=1e-5
    )
    np.save(os.path.join(test_dir, f'{test_str}_data.npy'), data)
    np.save(os.path.join(test_dir, f'{test_str}_result.npy'), result_.detach().numpy())
    np.save(os.path.join(test_dir, f'{test_str}_grad.npy'), x_.grad.detach().numpy())


@pytest.mark.ref_teacher_a2_1
@pytest.mark.parametrize("sizes", [(5,), (128,), (1, 64), (128, 256)])
@pytest.mark.parametrize("exp", [0, 1, 2, 3])
@pytest.mark.parametrize("backend", backend_tests)
def test_ref_teacher_pow_1(sizes, exp, backend):
    np.random.seed(20)
    test_dir = f'./tests/data_teacher/pow_1'
    test_str = str(exp)+'_' + '_'.join(map(str, sizes))
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    data = np.random.randn(*sizes)
    x = minitorch.tensor(data.tolist(), backend=shared[backend], requires_grad=True)
    x_ = torch.tensor(data, dtype=torch.float32, requires_grad=True)

    result = x ** exp
    result_ = x_ ** exp

    np.testing.assert_allclose(
        result.to_numpy(),
        result_.detach().numpy(),
        atol=1e-5, 
        rtol=1e-5
    )

    # Backward
    result.sum().backward()
    result_.sum().backward()

    np.testing.assert_allclose(
        x.grad.to_numpy(), 
        x_.grad.detach().numpy(), 
        atol=1e-5, 
        rtol=1e-5
    )
    np.save(os.path.join(test_dir, f'{test_str}_data.npy'), data)
    np.save(os.path.join(test_dir, f'{test_str}_result.npy'), result_.detach().numpy())
    np.save(os.path.join(test_dir, f'{test_str}_grad.npy'), x_.grad.detach().numpy())

################################ POW 2 ########################################

@pytest.mark.ref_student_a2_1
@pytest.mark.parametrize("sizes", [(5,), (128,), (1, 64), (128, 256)])
@pytest.mark.parametrize("exp", [0.5])
@pytest.mark.parametrize("backend", backend_tests)
def test_ref_student_pow_2(sizes, exp, backend):
    np.random.seed(10)
    test_dir = f'./tests/data/pow_2'
    test_str = str(exp)+'_' + '_'.join(map(str, sizes))
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    data = np.random.uniform(0, 1, size=sizes)
    x = minitorch.tensor(data.tolist(), backend=shared[backend], requires_grad=True)
    x_ = torch.tensor(data, dtype=torch.float32, requires_grad=True)

    result = x ** exp
    result_ = x_ ** exp

    np.testing.assert_allclose(
        result.to_numpy(),
        result_.detach().numpy(),
        atol=1e-5, 
        rtol=1e-5
    )

    # Backward
    result.sum().backward()
    result_.sum().backward()

    np.testing.assert_allclose(
        x.grad.to_numpy(), 
        x_.grad.detach().numpy(), 
        atol=1e-5, 
        rtol=1e-5
    )

    np.save(os.path.join(test_dir, f'{test_str}_data.npy'), data)
    np.save(os.path.join(test_dir, f'{test_str}_result.npy'), result_.detach().numpy())
    np.save(os.path.join(test_dir, f'{test_str}_grad.npy'), x_.grad.detach().numpy())


@pytest.mark.ref_teacher_a2_1
@pytest.mark.parametrize("sizes", [(5,), (128,), (1, 64), (128, 256)])
@pytest.mark.parametrize("exp", [0.5])
@pytest.mark.parametrize("backend", backend_tests)
def test_ref_teacher_pow_2(sizes, exp, backend):
    np.random.seed(20)
    test_dir = f'./tests/data_teacher/pow_2'
    test_str = str(exp)+'_' + '_'.join(map(str, sizes))
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    data = np.random.uniform(0, 1, size=sizes)
    x = minitorch.tensor(data.tolist(), backend=shared[backend], requires_grad=True)
    x_ = torch.tensor(data, dtype=torch.float32, requires_grad=True)

    result = x ** exp
    result_ = x_ ** exp

    np.testing.assert_allclose(
        result.to_numpy(),
        result_.detach().numpy(),
        atol=1e-5, 
        rtol=1e-5
    )

    # Backward
    result.sum().backward()
    result_.sum().backward()

    np.testing.assert_allclose(
        x.grad.to_numpy(), 
        x_.grad.detach().numpy(), 
        atol=1e-5, 
        rtol=1e-5
    )
    np.save(os.path.join(test_dir, f'{test_str}_data.npy'), data)
    np.save(os.path.join(test_dir, f'{test_str}_result.npy'), result_.detach().numpy())
    np.save(os.path.join(test_dir, f'{test_str}_grad.npy'), x_.grad.detach().numpy())

################################ POW 3 ########################################

@pytest.mark.ref_student_a2_1
@pytest.mark.parametrize("sizes", [(5,), (128,), (1, 64), (128, 256)])
@pytest.mark.parametrize("exp", [1, 2])
@pytest.mark.parametrize("backend", backend_tests)
def test_ref_student_pow_3(sizes, exp, backend):
    np.random.seed(10)
    test_dir = f'./tests/data/pow_3'
    test_str = str(exp)+'_' + '_'.join(map(str, sizes))
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    data = np.random.randn(*sizes)
    x = minitorch.tensor(data.tolist(), backend=shared[backend], requires_grad=True)
    x_ = torch.tensor(data, dtype=torch.float32, requires_grad=True)

    result = (x ** exp) * 1.34 
    result_ = (x_ ** exp) * 1.34 

    np.testing.assert_allclose(
        result.to_numpy(),
        result_.detach().numpy(),
        atol=1e-5, 
        rtol=1e-5
    )

    # Backward
    result.sum().backward()
    result_.sum().backward()

    np.testing.assert_allclose(
        x.grad.to_numpy(), 
        x_.grad.detach().numpy(), 
        atol=1e-5, 
        rtol=1e-5
    )
    np.save(os.path.join(test_dir, f'{test_str}_data.npy'), data)
    np.save(os.path.join(test_dir, f'{test_str}_result.npy'), result_.detach().numpy())
    np.save(os.path.join(test_dir, f'{test_str}_grad.npy'), x_.grad.detach().numpy())


@pytest.mark.ref_teacher_a2_1
@pytest.mark.parametrize("sizes", [(5,), (128,), (1, 64), (128, 256)])
@pytest.mark.parametrize("exp", [1, 2])
@pytest.mark.parametrize("backend", backend_tests)
def test_ref_teacher_pow_3(sizes, exp, backend):
    np.random.seed(20)
    test_dir = f'./tests/data_teacher/pow_3'
    test_str = str(exp)+'_' + '_'.join(map(str, sizes))
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    data = np.random.randn(*sizes)
    x = minitorch.tensor(data.tolist(), backend=shared[backend], requires_grad=True)
    x_ = torch.tensor(data, dtype=torch.float32, requires_grad=True)

    result = (x ** exp) * 1.34 
    result_ = (x_ ** exp) * 1.34 

    np.testing.assert_allclose(
        result.to_numpy(),
        result_.detach().numpy(),
        atol=1e-5, 
        rtol=1e-5
    )

    # Backward
    result.sum().backward()
    result_.sum().backward()

    np.testing.assert_allclose(
        x.grad.to_numpy(), 
        x_.grad.detach().numpy(), 
        atol=1e-5, 
        rtol=1e-5
    )
    np.save(os.path.join(test_dir, f'{test_str}_data.npy'), data)
    np.save(os.path.join(test_dir, f'{test_str}_result.npy'), result_.detach().numpy())
    np.save(os.path.join(test_dir, f'{test_str}_grad.npy'), x_.grad.detach().numpy())

################################ TANH 1 ########################################

@pytest.mark.ref_student_a2_1
@pytest.mark.parametrize("sizes", [(5,), (128,), (1, 64), (128, 256)])
@pytest.mark.parametrize("backend", backend_tests)
def test_ref_student_tanh_1(sizes, backend):
    np.random.seed(10)
    test_dir = f'./tests/data/tanh_1'
    test_str = '_'.join(map(str, sizes))
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    data = np.random.randn(*sizes)
    
    x = minitorch.tensor(data.tolist(), backend=shared[backend], requires_grad=True)
    x_ = torch.tensor(data, dtype=torch.float32, requires_grad=True)

    result = x.tanh()
    result_ = torch.tanh(x_)

    np.testing.assert_allclose(
        result.to_numpy(),
        result_.detach().numpy(),
        atol=1e-5, 
        rtol=1e-5
    )

    # Backward
    result.sum().backward()
    result_.sum().backward()

    np.testing.assert_allclose(
        x.grad.to_numpy(), 
        x_.grad.detach().numpy(), 
        atol=1e-5, 
        rtol=1e-5
    )

    np.save(os.path.join(test_dir, f'{test_str}_data.npy'), data)
    np.save(os.path.join(test_dir, f'{test_str}_result.npy'), result_.detach().numpy())
    np.save(os.path.join(test_dir, f'{test_str}_grad.npy'), x_.grad.detach().numpy())


@pytest.mark.ref_teacher_a2_1
@pytest.mark.parametrize("sizes", [(5,), (128,), (1, 64), (128, 256)])
@pytest.mark.parametrize("backend", backend_tests)
def test_ref_teacher_tanh_1(sizes, backend):
    np.random.seed(10)
    test_dir = f'./tests/data_teacher/tanh_1'
    test_str = '_'.join(map(str, sizes))
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    data = np.random.randn(*sizes)
    
    x = minitorch.tensor(data.tolist(), backend=shared[backend], requires_grad=True)
    x_ = torch.tensor(data, dtype=torch.float32, requires_grad=True)

    result = x.tanh()
    result_ = torch.tanh(x_)

    np.testing.assert_allclose(
        result.to_numpy(),
        result_.detach().numpy(),
        atol=1e-5, 
        rtol=1e-5
    )

    # Backward
    result.sum().backward()
    result_.sum().backward()

    np.testing.assert_allclose(
        x.grad.to_numpy(), 
        x_.grad.detach().numpy(), 
        atol=1e-5, 
        rtol=1e-5
    )

    np.save(os.path.join(test_dir, f'{test_str}_data.npy'), data)
    np.save(os.path.join(test_dir, f'{test_str}_result.npy'), result_.detach().numpy())
    np.save(os.path.join(test_dir, f'{test_str}_grad.npy'), x_.grad.detach().numpy())

################################ TANH 2 ########################################

@pytest.mark.ref_student_a2_1
@pytest.mark.parametrize("sizes", [(5,), (128,), (1, 64), (128, 256)])
@pytest.mark.parametrize("backend", backend_tests)
def test_ref_student_tanh_2(sizes, backend):
    np.random.seed(20)
    test_dir = f'./tests/data/tanh_2'
    test_str = '_'.join(map(str, sizes))
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    data = np.random.randn(*sizes)
    
    x = minitorch.tensor(data.tolist(), backend=shared[backend], requires_grad=True)
    x_ = torch.tensor(data, dtype=torch.float32, requires_grad=True)

    result = (x.tanh() + 1.0) * 0.5
    result_ = (torch.tanh(x_) + 1.0) * 0.5

    np.testing.assert_allclose(
        result.to_numpy(),
        result_.detach().numpy(),
        atol=1e-5, 
        rtol=1e-5
    )

    # Backward
    result.sum().backward()
    result_.sum().backward()

    np.testing.assert_allclose(
        x.grad.to_numpy(), 
        x_.grad.detach().numpy(), 
        atol=1e-5, 
        rtol=1e-5
    )

    np.save(os.path.join(test_dir, f'{test_str}_data.npy'), data)
    np.save(os.path.join(test_dir, f'{test_str}_result.npy'), result_.detach().numpy())
    np.save(os.path.join(test_dir, f'{test_str}_grad.npy'), x_.grad.detach().numpy())


@pytest.mark.ref_teacher_a2_1
@pytest.mark.parametrize("sizes", [(5,), (128,), (1, 64), (128, 256)])
@pytest.mark.parametrize("backend", backend_tests)
def test_ref_teacher_tanh_2(sizes, backend):
    np.random.seed(20)
    test_dir = f'./tests/data_teacher/tanh_2'
    test_str = '_'.join(map(str, sizes))
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    data = np.random.randn(*sizes)
    
    x = minitorch.tensor(data.tolist(), backend=shared[backend], requires_grad=True)
    x_ = torch.tensor(data, dtype=torch.float32, requires_grad=True)

    result = (x.tanh() + 1.0) * 0.5
    result_ = (torch.tanh(x_) + 1.0) * 0.5

    np.testing.assert_allclose(
        result.to_numpy(),
        result_.detach().numpy(),
        atol=1e-5, 
        rtol=1e-5
    )

    # Backward
    result.sum().backward()
    result_.sum().backward()

    np.testing.assert_allclose(
        x.grad.to_numpy(), 
        x_.grad.detach().numpy(), 
        atol=1e-5, 
        rtol=1e-5
    )

    np.save(os.path.join(test_dir, f'{test_str}_data.npy'), data)
    np.save(os.path.join(test_dir, f'{test_str}_result.npy'), result_.detach().numpy())
    np.save(os.path.join(test_dir, f'{test_str}_grad.npy'), x_.grad.detach().numpy())
import pytest
import minitorch
from minitorch.cuda_kernel_ops import CudaKernelOps

import numpy as np
import torch
import torch.nn as nn
import numba
import os

np.random.seed(3)


_BACKENDS = [pytest.param(
                 minitorch.TensorBackend(CudaKernelOps), 
                 marks=pytest.mark.skipif(not numba.cuda.is_available(), reason="No GPU")
             )]

################################ EMBEDDING ########################################

@pytest.mark.ref_student_a2_3
@pytest.mark.parametrize("batch_size", [1, 5])
@pytest.mark.parametrize("num_embeddings", [3, 200])
@pytest.mark.parametrize("seq_len", [1, 50])
@pytest.mark.parametrize("embedding_dim", [256])
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_ref_student_embedding(batch_size, num_embeddings, seq_len, embedding_dim, backend):
    np.random.seed(10)
    torch.manual_seed(10)
    test_dir = f'./tests/data/embedding'
    test_str = '_'.join(map(str, (batch_size, num_embeddings, seq_len, embedding_dim)))
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    data = np.random.randint(0, num_embeddings, size=(batch_size, seq_len))
    X = minitorch.tensor_from_numpy(data, backend=backend)
    X_ = torch.tensor(data)

    layer_ = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

    layer = minitorch.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim, backend=backend)
    layer.weights.value = minitorch.tensor_from_numpy(layer_.weight.detach().numpy(), backend=backend, requires_grad=True)

    result = layer(X)
    result_ = layer_(X_)

    np.testing.assert_allclose(
        result.to_numpy(),
        result_.detach().numpy(),
        atol=1e-5, 
        rtol=1e-5
    )

    result.sum().backward()
    result_.sum().backward()

    np.testing.assert_allclose(
        layer.weights.value.grad.to_numpy(),
        layer_.weight.grad.detach().numpy(), 
        atol=1e-5, 
        rtol=1e-5
    )

    np.save(os.path.join(test_dir, f'{test_str}_data.npy'), data)
    np.save(os.path.join(test_dir, f'{test_str}_layer_weight.npy'), layer_.weight.detach().numpy())
    np.save(os.path.join(test_dir, f'{test_str}_result.npy'), result_.detach().numpy())
    np.save(os.path.join(test_dir, f'{test_str}_weight_grad.npy'), layer_.weight.grad.detach().numpy())


@pytest.mark.ref_teacher_a2_3
@pytest.mark.parametrize("batch_size", [1, 5])
@pytest.mark.parametrize("num_embeddings", [3, 200])
@pytest.mark.parametrize("seq_len", [1, 50])
@pytest.mark.parametrize("embedding_dim", [256])
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_ref_teacher_embedding(batch_size, num_embeddings, seq_len, embedding_dim, backend):
    np.random.seed(20)
    torch.manual_seed(20)
    test_dir = f'./tests/data_teacher/embedding'
    test_str = '_'.join(map(str, (batch_size, num_embeddings, seq_len, embedding_dim)))
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    data = np.random.randint(0, num_embeddings, size=(batch_size, seq_len))
    X = minitorch.tensor_from_numpy(data, backend=backend)
    X_ = torch.tensor(data)

    layer_ = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

    layer = minitorch.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim, backend=backend)
    layer.weights.value = minitorch.tensor_from_numpy(layer_.weight.detach().numpy(), backend=backend, requires_grad=True)

    result = layer(X)
    result_ = layer_(X_)

    np.testing.assert_allclose(
        result.to_numpy(),
        result_.detach().numpy(),
        atol=1e-5, 
        rtol=1e-5
    )

    result.sum().backward()
    result_.sum().backward()

    np.testing.assert_allclose(
        layer.weights.value.grad.to_numpy(),
        layer_.weight.grad.detach().numpy(), 
        atol=1e-5, 
        rtol=1e-5
    )

    np.save(os.path.join(test_dir, f'{test_str}_data.npy'), data)
    np.save(os.path.join(test_dir, f'{test_str}_layer_weight.npy'), layer_.weight.detach().numpy())
    np.save(os.path.join(test_dir, f'{test_str}_result.npy'), result_.detach().numpy())
    np.save(os.path.join(test_dir, f'{test_str}_weight_grad.npy'), layer_.weight.grad.detach().numpy())


@pytest.mark.ref_student_a2_3
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_ref_student_dropout(backend):
    np.random.seed(10)
    test_dir = f'./tests/data/dropout'
    test_str = 'dropout'
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    # Dropout ratio 0 means nothing gets deleted 
    data = np.random.randn(10, 10)
    x = minitorch.tensor(data.tolist(), backend=backend)
    layer = minitorch.Dropout(p_dropout=0)
    result = layer(x)
    np.testing.assert_allclose(result.to_numpy(), data, atol=1e-5, rtol=1e-5)

    # Nothing should be dropped when not training
    layer = minitorch.Dropout(p_dropout=0.5)
    layer.training = False
    result = layer(x)
    np.testing.assert_allclose(result.to_numpy(), data, atol=1e-5, rtol=1e-5)

    layer = minitorch.Dropout(p_dropout = 0.5)
    layer.training = True
    ref_sol = layer(x).to_numpy()

    np.save(os.path.join(test_dir, f'{test_str}.npy'), ref_sol)


@pytest.mark.ref_teacher_a2_3
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_ref_teacher_dropout(backend):
    np.random.seed(20)
    test_dir = f'./tests/data_teacher/dropout'
    test_str = 'dropout'
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    # Dropout ratio 0 means nothing gets deleted 
    data = np.random.randn(10, 10)
    x = minitorch.tensor(data.tolist(), backend=backend)
    layer = minitorch.Dropout(p_dropout=0)
    result = layer(x)
    np.testing.assert_allclose(result.to_numpy(), data, atol=1e-5, rtol=1e-5)

    # Nothing should be dropped when not training
    layer = minitorch.Dropout(p_dropout=0.5)
    layer.training = False
    result = layer(x)
    np.testing.assert_allclose(result.to_numpy(), data, atol=1e-5, rtol=1e-5)

    layer = minitorch.Dropout(p_dropout = 0.5)
    layer.training = True
    ref_sol = layer(x).to_numpy()

    np.save(os.path.join(test_dir, f'{test_str}.npy'), ref_sol)


@pytest.mark.ref_student_a2_3
@pytest.mark.parametrize("sizes", [(64, 256, 128), (8, 256, 8), (128, 256, 512)])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_ref_student_linear(sizes, bias, backend):
    np.random.seed(10)
    torch.manual_seed(10)
    test_dir = f'./tests/data/linear'
    test_str = '_'.join(map(str, sizes + (bias,)))
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    
    m, n, p = sizes
    data = np.random.randn(m, n)
    X = minitorch.tensor_from_numpy(data, backend, True)
    X_ = torch.tensor(data, dtype=torch.float32, requires_grad=True)

    layer_ = torch.nn.Linear(in_features=n, out_features=p, bias=bias, dtype=torch.float32)
    layer = minitorch.Linear(in_size=n, out_size=p, bias=bias, backend=backend)

    weights = layer_.weight.detach().numpy().T
    layer.weights.value = minitorch.tensor_from_numpy(weights.copy(), backend, requires_grad=True)
    if bias:    
        b = layer_.bias.detach().numpy()
        layer.bias.value = minitorch.tensor_from_numpy(b.copy(), backend, requires_grad=True)
    
    result = layer(X)
    result_ = layer_(X_)

    np.testing.assert_allclose(
        result.to_numpy(),
        result_.detach().numpy(), 
        rtol=1e-5,
        atol=1e-5
    )

    result.sum().backward()
    result_.sum().backward()

    np.testing.assert_allclose(
        X.grad.to_numpy(),
        X_.grad.detach().numpy(), 
        rtol=1e-5,
        atol=1e-5
    )

    np.testing.assert_allclose(
        layer.weights.value.grad.to_numpy(),
        layer_.weight.grad.detach().numpy().T, 
        rtol=1e-5,
        atol=1e-5
    )

    if bias:
        np.testing.assert_allclose(
            layer.bias.value.grad.to_numpy(),
            layer_.bias.grad.detach().numpy(), 
            rtol=1e-5,
            atol=1e-5
        )
    
    np.save(os.path.join(test_dir, f'{test_str}_data.npy'), data)
    np.save(os.path.join(test_dir, f'{test_str}_layer_weight.npy'), layer_.weight.detach().numpy().T)
    np.save(os.path.join(test_dir, f'{test_str}_result.npy'), result_.detach().numpy())
    np.save(os.path.join(test_dir, f'{test_str}_weight_grad.npy'), layer_.weight.grad.detach().numpy().T)
    np.save(os.path.join(test_dir, f'{test_str}_X_grad.npy'), X_.grad.detach().numpy())
    if bias:
        np.save(os.path.join(test_dir, f'{test_str}_layer_bias.npy'), layer_.bias.detach().numpy())
        np.save(os.path.join(test_dir, f'{test_str}_bias_grad.npy'), layer_.bias.grad.detach().numpy())


@pytest.mark.ref_teacher_a2_3
@pytest.mark.parametrize("sizes", [(64, 256, 128), (8, 256, 8), (128, 256, 512)])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_ref_teacher_linear(sizes, bias, backend):
    np.random.seed(20)
    torch.manual_seed(20)
    test_dir = f'./tests/data_teacher/linear'
    test_str = '_'.join(map(str, sizes + (bias,)))
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    
    m, n, p = sizes
    data = np.random.randn(m, n)
    X = minitorch.tensor_from_numpy(data, backend, True)
    X_ = torch.tensor(data, dtype=torch.float32, requires_grad=True)

    layer_ = torch.nn.Linear(in_features=n, out_features=p, bias=bias, dtype=torch.float32)
    layer = minitorch.Linear(in_size=n, out_size=p, bias=bias, backend=backend)

    weights = layer_.weight.detach().numpy().T
    layer.weights.value = minitorch.tensor_from_numpy(weights.copy(), backend, requires_grad=True)
    if bias:    
        b = layer_.bias.detach().numpy()
        layer.bias.value = minitorch.tensor_from_numpy(b.copy(), backend, requires_grad=True)
    
    result = layer(X)
    result_ = layer_(X_)

    np.testing.assert_allclose(
        result.to_numpy(),
        result_.detach().numpy(), 
        rtol=1e-5,
        atol=1e-5
    )

    result.sum().backward()
    result_.sum().backward()

    np.testing.assert_allclose(
        X.grad.to_numpy(),
        X_.grad.detach().numpy(), 
        rtol=1e-5,
        atol=1e-5
    )

    np.testing.assert_allclose(
        layer.weights.value.grad.to_numpy(),
        layer_.weight.grad.detach().numpy().T, 
        rtol=1e-5,
        atol=1e-5
    )

    if bias:
        np.testing.assert_allclose(
            layer.bias.value.grad.to_numpy(),
            layer_.bias.grad.detach().numpy(), 
            rtol=1e-5,
            atol=1e-5
        )
    
    np.save(os.path.join(test_dir, f'{test_str}_data.npy'), data)
    np.save(os.path.join(test_dir, f'{test_str}_layer_weight.npy'), layer_.weight.detach().numpy().T)
    np.save(os.path.join(test_dir, f'{test_str}_result.npy'), result_.detach().numpy())
    np.save(os.path.join(test_dir, f'{test_str}_weight_grad.npy'), layer_.weight.grad.detach().numpy().T)
    np.save(os.path.join(test_dir, f'{test_str}_X_grad.npy'), X_.grad.detach().numpy())
    if bias:
        np.save(os.path.join(test_dir, f'{test_str}_layer_bias.npy'), layer_.bias.detach().numpy())
        np.save(os.path.join(test_dir, f'{test_str}_bias_grad.npy'), layer_.bias.grad.detach().numpy())


@pytest.mark.parametrize("sizes", [(64, 128, 256)])
@pytest.mark.parametrize("bias", [False])
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_linear_double(sizes, bias, backend):
    ## FOR FFN 
    np.random.seed(10)
    torch.manual_seed(10)
    
    bs, n_embd, middle_dim = sizes
    data = np.random.randn(bs, n_embd) * 20
    X = minitorch.tensor_from_numpy(data, backend, True)
    X_ = torch.tensor(data, dtype=torch.float32, requires_grad=True)

    layer_1_ = torch.nn.Linear(in_features=n_embd, out_features=middle_dim, bias=bias, dtype=torch.float32)
    layer_2_ = torch.nn.Linear(in_features=middle_dim, out_features=n_embd, bias=bias, dtype=torch.float32)
    layer_1 = minitorch.Linear(in_size=n_embd, out_size=middle_dim, bias=bias, backend=backend)
    layer_2 = minitorch.Linear(in_size=middle_dim, out_size=n_embd, bias=bias, backend=backend)

    weights_1 = layer_1_.weight.detach().numpy().T
    weights_2 = layer_2_.weight.detach().numpy().T
    layer_1.weights.value = minitorch.tensor_from_numpy(weights_1.copy(), backend, requires_grad=True)
    layer_2.weights.value = minitorch.tensor_from_numpy(weights_2.copy(), backend, requires_grad=True)
    if bias:    
        b_1 = layer_1_.bias.detach().numpy()
        layer_1.bias.value = minitorch.tensor_from_numpy(b_1.copy(), backend, requires_grad=True)
        b_2 = layer_2_.bias.detach().numpy()
        layer_2.bias.value = minitorch.tensor_from_numpy(b_2.copy(), backend, requires_grad=True)
    
    result = layer_2(layer_1(X))
    result_ = layer_2_(layer_1_(X_))

    np.testing.assert_allclose(
        result.to_numpy(),
        result_.detach().numpy(), 
        rtol=1e-5,
        atol=1e-5
    )

    result.sum().backward()
    result_.sum().backward()

    np.testing.assert_allclose(
        X.grad.to_numpy(),
        X_.grad.detach().numpy(), 
        rtol=1e-5,
        atol=1e-5
    )

    np.testing.assert_allclose(
        layer_1.weights.value.grad.to_numpy(),
        layer_1_.weight.grad.detach().numpy().T, 
        rtol=1e-5,
        atol=1e-5
    )
    np.testing.assert_allclose(
        layer_2.weights.value.grad.to_numpy(),
        layer_2_.weight.grad.detach().numpy().T, 
        rtol=1e-5,
        atol=1e-5
    )

    if bias:
        np.testing.assert_allclose(
            layer_1.bias.value.grad.to_numpy(),
            layer_1_.bias.grad.detach().numpy(), 
            rtol=1e-5,
            atol=1e-5
        )
        np.testing.assert_allclose(
            layer_2.bias.value.grad.to_numpy(),
            layer_2_.bias.grad.detach().numpy(), 
            rtol=1e-5,
            atol=1e-5
        )


@pytest.mark.ref_student_a2_3
@pytest.mark.parametrize("batch_size", [1, 64])
@pytest.mark.parametrize("dim", [3, 128, 256])
@pytest.mark.parametrize("eps", [1e-5])
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_ref_student_layernorm(batch_size, dim, eps, backend):
    np.random.seed(10)
    torch.manual_seed(10)
    test_dir = f'./tests/data/layernorm'
    test_str = '_'.join(map(str, (batch_size, dim)))
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    data = np.random.randn(batch_size, dim)

    layer = minitorch.LayerNorm1d(
        dim=dim, eps=eps, backend=backend
    )

    layer_ = torch.nn.LayerNorm(
        normalized_shape=dim, eps=eps
    )
    x_minitorch = minitorch.tensor(data.tolist(), backend=backend)
    x_torch = torch.tensor(data.tolist(), dtype=torch.float32, requires_grad=True)

    result = layer(x_minitorch)
    result_ = layer_(x_torch)

    np.testing.assert_allclose(
        result.to_numpy(),
        result_.detach().numpy(),
        atol=1e-5, 
        rtol=1e-5
    )

    result.sum().backward()
    result_.sum().backward()

    np.testing.assert_allclose(
        x_minitorch.grad.to_numpy(),
        x_torch.grad.detach().numpy(),
        atol=1e-5,
        rtol=1e-5
    )

    np.testing.assert_allclose(
        layer.weights.value.grad.to_numpy(),
        layer_.weight.grad.detach().numpy(),
        atol=1e-5,
        rtol=1e-5
    )

    np.save(os.path.join(test_dir, f'{test_str}_data.npy'), data)
    np.save(os.path.join(test_dir, f'{test_str}_result.npy'), result_.detach().numpy())
    np.save(os.path.join(test_dir, f'{test_str}_x_grad.npy'), x_torch.grad.detach().numpy())
    np.save(os.path.join(test_dir, f'{test_str}_weight_grad.npy'), layer_.weight.grad.detach().numpy())


@pytest.mark.ref_teacher_a2_3
@pytest.mark.parametrize("batch_size", [1, 64])
@pytest.mark.parametrize("dim", [3, 128, 256])
@pytest.mark.parametrize("eps", [1e-5])
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_ref_teacher_layernorm(batch_size, dim, eps, backend):
    np.random.seed(20)
    torch.manual_seed(20)
    test_dir = f'./tests/data_teacher/layernorm'
    test_str = '_'.join(map(str, (batch_size, dim)))
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    data = np.random.randn(batch_size, dim)

    layer = minitorch.LayerNorm1d(
        dim=dim, eps=eps, backend=backend
    )

    layer_ = torch.nn.LayerNorm(
        normalized_shape=dim, eps=eps
    )
    x_minitorch = minitorch.tensor(data.tolist(), backend=backend)
    x_torch = torch.tensor(data.tolist(), dtype=torch.float32, requires_grad=True)

    result = layer(x_minitorch)
    result_ = layer_(x_torch)

    np.testing.assert_allclose(
        result.to_numpy(),
        result_.detach().numpy(),
        atol=1e-5, 
        rtol=1e-5
    )

    result.sum().backward()
    result_.sum().backward()

    np.testing.assert_allclose(
        x_minitorch.grad.to_numpy(),
        x_torch.grad.detach().numpy(),
        atol=1e-5,
        rtol=1e-5
    )

    np.testing.assert_allclose(
        layer.weights.value.grad.to_numpy(),
        layer_.weight.grad.detach().numpy(),
        atol=1e-5,
        rtol=1e-5
    )

    np.save(os.path.join(test_dir, f'{test_str}_data.npy'), data)
    np.save(os.path.join(test_dir, f'{test_str}_result.npy'), result_.detach().numpy())
    np.save(os.path.join(test_dir, f'{test_str}_x_grad.npy'), x_torch.grad.detach().numpy())
    np.save(os.path.join(test_dir, f'{test_str}_weight_grad.npy'), layer_.weight.grad.detach().numpy())
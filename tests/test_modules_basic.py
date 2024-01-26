import pytest
import minitorch
from minitorch.cuda_kernel_ops import CudaKernelOps

import numpy as np
import torch
import torch.nn as nn
import numba

np.random.seed(3)


_BACKENDS = [pytest.param(
                 minitorch.TensorBackend(CudaKernelOps), 
                 marks=pytest.mark.skipif(not numba.cuda.is_available(), reason="No GPU")
             )]

@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("num_embeddings", [3, 20])
@pytest.mark.parametrize("seq_len", [1, 5])
@pytest.mark.parametrize("embedding_dim", [5])
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_embedding(batch_size, num_embeddings, seq_len, embedding_dim, backend):
    np.random.seed(11868)

    x = np.random.randint(0, num_embeddings, size=(batch_size, seq_len))

    layer = minitorch.Embedding(
        num_embeddings=num_embeddings, embedding_dim=embedding_dim, backend=backend
    )

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer_ = torch.nn.Embedding(
        num_embeddings=num_embeddings, embedding_dim=embedding_dim,
    )
    
    # Reset weights
    layer.weights.value = minitorch.tensor(layer_.weight.detach().numpy().tolist(), backend=backend)

    result = layer(minitorch.tensor(x.tolist(), backend=backend))
    result_ = layer_(torch.tensor(x))

    np.testing.assert_allclose(
        result.to_numpy(),
        result_.detach().numpy(),
        atol=1e-5, 
        rtol=1e-5
    )

    a = minitorch.tensor(x.tolist(), backend=backend)

    assert result is not None

def test_dropout():
    pass

def test_linear():
    pass

@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("dim", [3])
@pytest.mark.parametrize("eps", [1e-5])
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_layernorm(batch_size, dim, eps, backend):
    np.random.seed(5)

    x = np.random.randn(batch_size, dim)

    layer = minitorch.LayerNorm1d(
        dim=dim, eps=eps, backend=backend
    )

    layer_ = torch.nn.LayerNorm(
        normalized_shape=dim, eps=eps
    )
    x_minitorch = minitorch.tensor(x.tolist(), backend=backend)
    x_torch = torch.tensor(x.tolist(), requires_grad=True)

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
        x_torch.grad.detach().numpy()
    )

    # minitorch.grad_check(lambda x: layer(x), minitorch.tensor(x.tolist(), backend=backend))

    assert result is not None
import pytest
import minitorch
from minitorch.cuda_kernel_ops import CudaKernelOps

import numpy as np
import torch
import torch.nn as nn
import numba

np.random.seed(3)

datatype = np.float32

_BACKENDS = [pytest.param(
                 minitorch.TensorBackend(CudaKernelOps), 
                 marks=pytest.mark.skipif(not numba.cuda.is_available(), reason="No GPU")
             )]

@pytest.mark.parametrize("batch_size",  [1, 5])
@pytest.mark.parametrize("queries_len", [2, 10])
@pytest.mark.parametrize("n_embd",      [64, 128])
@pytest.mark.parametrize("num_heads",   [1, 4])
@pytest.mark.parametrize("p_dropout",   [0.0])
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_multihead_attention(batch_size, queries_len, n_embd, num_heads, p_dropout, backend):
    np.random.seed(10)
    torch.manual_seed(10)

    data = np.random.rand(batch_size, queries_len, n_embd)
    X    = minitorch.tensor_from_numpy(data, backend, True)
    X_   = torch.tensor(data, dtype=torch.float32, requires_grad=True)

    layer_ = torch.nn.MultiheadAttention(n_embd, num_heads, p_dropout, bias=False, batch_first=True, dtype=torch.float32)
    layer = minitorch.MultiHeadAttention(n_embd, num_heads, True, p_dropout, bias=False, backend=backend)
    
    # Set weights of minitorch layer to torch weights
    w_qkv = layer_.in_proj_weight.detach().numpy().T.copy() # (n_embd, 3*n_embd)
    w_q_, w_k_, w_v_ = [w.copy() for w in np.split(w_qkv, 3, -1)] # 3 * (n_embd, n_embd)
    w_out_ = layer_.out_proj.weight.detach().numpy().T.copy()

    """
    IF I DO .CONTIGUOUS, IT DOES A COPY WHICH IS WHY NONE OF THESE WORK BS LAST FN IS NOT NONE
    """

    w_q = minitorch.tensor_from_numpy((w_q_), backend=backend, requires_grad=True)
    w_k = minitorch.tensor_from_numpy((w_k_), backend=backend, requires_grad=True)
    w_v = minitorch.tensor_from_numpy((w_v_), backend=backend, requires_grad=True)
    w_out = minitorch.tensor_from_numpy((w_out_), backend=backend, requires_grad=True)

    layer.q_projection.weights.value = w_q
    layer.k_projection.weights.value = w_k
    layer.v_projection.weights.value = w_v
    layer.out_projection.weights.value = w_out

    # The same mask is causal mask
    M = torch.triu(-float("inf")*torch.ones(queries_len, queries_len),1)

    result = layer(X)
    result_, _ = layer_(X_, X_, X_, attn_mask = M)
    
    np.testing.assert_allclose(
        result.to_numpy(),
        result_.detach().numpy(),
        atol=1e-5,
        rtol=1e-5
    )

    # Check backward
    result.sum().backward()
    result_.sum().backward()
    
    np.testing.assert_allclose(
        X.grad.to_numpy(),
        X_.grad.detach().numpy(),
        atol=1e-5,
        rtol=1e-5
    )

    np.testing.assert_allclose(
        layer.out_projection.weights.value.grad.to_numpy(),
        layer_.out_proj.weight.grad.detach().numpy().T,
        atol=1e-5,
        rtol=1e-5
    )

    # Since the torch W_Q, W_K, W_V is all one matrix, we can't compare
    assert (
        (layer.q_projection.weights.value.grad is not None) and
        (layer.k_projection.weights.value.grad is not None) and
        (layer.v_projection.weights.value.grad is not None)
    )


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("seq_len", [5])
@pytest.mark.parametrize("n_embd",  [9])
@pytest.mark.parametrize("dropout", [0.0])
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_feedforward_layer(batch_size, seq_len, n_embd, dropout, backend):

    np.random.seed(19943)

    x = np.random.randn(
        batch_size, seq_len, n_embd
    ).astype(datatype)

    layer = minitorch.FeedForward(
        n_embd=n_embd, p_dropout=dropout, bias=True, backend=backend)

    result = layer(
        minitorch.tensor(x.tolist(), backend=backend)
    )

    assert result is not None

@pytest.mark.parametrize("batch_size" [1, 5])
@pytest.mark.parametrize("seq_len",   [2, 10])
@pytest.mark.parametrize("n_embd",    [64, 128])
@pytest.mark.parametrize("num_heads", [1, 4])
@pytest.mark.parametrize("causal",    [True])
@pytest.mark.parametrize("p_dropout", [0.0])
@pytest.mark.parametrize("ln_eps", [1e-5])
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_transformer_layer(batch_size, seq_len, n_embd, num_heads, causal, dropout, ln_eps, bias, backend):

    np.random.seed(19943)

    x = np.random.randn(
        batch_size, seq_len, n_embd
    ).astype(datatype)

    layer = minitorch.TransformerLayer(
        n_embd=n_embd, n_head=num_heads, p_dropout=dropout, ln_eps=ln_eps, 
        bias=bias, backend=backend)
    
    layer_ = torch.nn.TransformerEncoderLayer(
        d_model=n_embd, nhead=num_heads, dim_feedforward=256, dropout=0,
        activation=lambda x: torch.nn.functional.gelu(x, approximate='tanh'),
        batch_first=True, dtype=torch.float32
    )

    result = layer(
        minitorch.tensor(x.tolist(), backend=backend)
    )

    assert result is not None


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("seq_len", [5])
@pytest.mark.parametrize("n_vocab", [10])
@pytest.mark.parametrize("n_embd",  [9])
@pytest.mark.parametrize("n_head",  [3])
@pytest.mark.parametrize("n_positions", [10])
@pytest.mark.parametrize("n_layer", [1])
@pytest.mark.parametrize("dropout", [0.0])
@pytest.mark.parametrize("ln_eps", [1e-5])
@pytest.mark.parametrize("bias", [True])
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_decoder_lm(batch_size, seq_len, n_vocab, n_embd, n_head, n_layer, n_positions, dropout, ln_eps, bias, backend):

    np.random.seed(19943)

    x = np.random.randint(
        low=0, high=n_vocab, size=(batch_size, seq_len)
    ).astype(datatype)

    layer = minitorch.DecoderLM(
        n_vocab=n_vocab, n_embd=n_embd, n_head=n_head, n_positions=n_positions, 
        n_layer=n_layer, p_dropout=dropout, ln_eps=ln_eps, bias=bias, backend=backend)

    result = layer(
        minitorch.tensor(x.tolist(), backend=backend)
    )

    assert result is not None
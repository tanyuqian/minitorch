import pytest
import minitorch
from minitorch.cuda_kernel_ops import CudaKernelOps

import numpy as np
import torch
import torch.nn as nn
import numba
import os

np.random.seed(3)

datatype = np.float32

_BACKENDS = [pytest.param(
                 minitorch.TensorBackend(CudaKernelOps), 
                 marks=pytest.mark.skipif(not numba.cuda.is_available(), reason="No GPU")
             )]

################################ DEBUGGING ########################################

@pytest.mark.parametrize("batch_size",  [1])
@pytest.mark.parametrize("queries_len", [2])
@pytest.mark.parametrize("n_embd",      [3])
@pytest.mark.parametrize("num_heads",   [1])
@pytest.mark.parametrize("p_dropout",   [0.0])
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_debug_multihead_attention(batch_size, queries_len, n_embd, num_heads, p_dropout, backend):
    np.random.seed(10)
    torch.manual_seed(10)

    data = np.random.randn(batch_size, queries_len, n_embd) * 10
    X    = minitorch.tensor_from_numpy(data, backend, True)
    X_   = torch.tensor(data, dtype=torch.float32, requires_grad=True)

    layer_ = torch.nn.MultiheadAttention(n_embd, num_heads, p_dropout, bias=False, batch_first=True, dtype=torch.float32)
    # layer = minitorch.MultiHeadAttention(n_embd, num_heads, True, p_dropout, bias=False, backend=backend)
    layer = minitorch.MultiHeadAttention(n_embd, num_heads, True, p_dropout, bias=False, backend=backend)
    
    # Set weights of minitorch layer to torch weights
    w_qkv = layer_.in_proj_weight.detach().numpy().T.copy() # (n_embd, 3*n_embd)
    w_q_, w_k_, w_v_ = [w.copy() for w in np.split(w_qkv, 3, -1)] # 3 * (n_embd, n_embd)
    w_out_ = layer_.out_proj.weight.detach().numpy().T.copy()

    w_q = minitorch.tensor_from_numpy((w_q_), backend=backend, requires_grad=True)
    w_k = minitorch.tensor_from_numpy((w_k_), backend=backend, requires_grad=True)
    w_v = minitorch.tensor_from_numpy((w_v_), backend=backend, requires_grad=True)
    w_out = minitorch.tensor_from_numpy((w_out_), backend=backend, requires_grad=True)

    layer.q_projection.weights.value = w_q
    layer.k_projection.weights.value = w_k
    layer.v_projection.weights.value = w_v
    layer.out_projection.weights.value = w_out

    # The same mask is causal mask
    mask = -np.finfo(datatype).max * np.triu(np.ones((queries_len, queries_len), dtype=datatype), 1)
    M = torch.tensor(mask, dtype=torch.float32)

    result = layer(X)
    result_, _ = layer_(X_, X_, X_, attn_mask = M)
    # result_, _ = layer_(X_, X_, X_)
    
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

    diff = abs(layer.out_projection.weights.value.grad.to_numpy() - layer_.out_proj.weight.grad.detach().numpy().T)
    print('max diff:', np.argmax(diff, axis=0))
    print('max diff:', np.argmax(diff, axis=1))

    np.testing.assert_allclose(
        layer.out_projection.weights.value.grad.to_numpy(),
        layer_.out_proj.weight.grad.detach().numpy().T,
        atol=1e-5,
        rtol=1e-5
    )

    w_qkv_grad = layer_.in_proj_weight.grad.detach().numpy().T.copy() # (n_embd, 3*n_embd)
    w_q_grad, w_k_grad, w_v_grad = [w.copy() for w in np.split(w_qkv_grad, 3, -1)] # 3 * (n_embd, n_embd)


    print("NUM PARAMS: ", len(layer.parameters()))

    np.testing.assert_allclose(
        layer.q_projection.weights.value.grad.to_numpy(),
        w_q_grad,
        atol=1e-5,
        rtol=1e-5
    )

    np.testing.assert_allclose(
        layer.k_projection.weights.value.grad.to_numpy(),
        w_k_grad,
        atol=1e-5,
        rtol=1e-5
    )
    np.testing.assert_allclose(
        layer.v_projection.weights.value.grad.to_numpy(),
        w_v_grad,
        atol=1e-5,
        rtol=1e-5
    )

################################ MULTIHEADATTENTION ########################################

@pytest.mark.ref_student_a2_4
@pytest.mark.parametrize("batch_size",  [1, 128])
@pytest.mark.parametrize("queries_len", [32, 40])
@pytest.mark.parametrize("n_embd",      [64, 256])
@pytest.mark.parametrize("num_heads",   [1, 4, 8])
@pytest.mark.parametrize("p_dropout",   [0.0])
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_ref_student_multihead_attention(batch_size, queries_len, n_embd, num_heads, p_dropout, backend):
    np.random.seed(10)
    torch.manual_seed(10)

    test_dir = f'./tests/data/multihead_attention'
    test_str = '_'.join(map(str, (batch_size, queries_len, n_embd, num_heads)))
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    # data = np.random.randn(batch_size, queries_len, n_embd)
    data = np.random.rand(batch_size, queries_len, n_embd) # NOTE: rand works, not randn
    X    = minitorch.tensor_from_numpy(data, backend, True)
    X_   = torch.tensor(data, dtype=torch.float32, requires_grad=True)

    layer_ = torch.nn.MultiheadAttention(n_embd, num_heads, p_dropout, bias=False, batch_first=True, dtype=torch.float32)
    layer = minitorch.MultiHeadAttention(n_embd, num_heads, True, p_dropout, bias=False, backend=backend)
    
    # Set weights of minitorch layer to torch weights
    w_qkv = layer_.in_proj_weight.detach().numpy().T.copy() # (n_embd, 3*n_embd)
    w_q_, w_k_, w_v_ = [w.copy() for w in np.split(w_qkv, 3, -1)] # 3 * (n_embd, n_embd)
    w_out_ = layer_.out_proj.weight.detach().numpy().T.copy()

    w_q = minitorch.tensor_from_numpy((w_q_), backend=backend, requires_grad=True)
    w_k = minitorch.tensor_from_numpy((w_k_), backend=backend, requires_grad=True)
    w_v = minitorch.tensor_from_numpy((w_v_), backend=backend, requires_grad=True)
    w_out = minitorch.tensor_from_numpy((w_out_), backend=backend, requires_grad=True)

    layer.q_projection.weights.value = w_q
    layer.k_projection.weights.value = w_k
    layer.v_projection.weights.value = w_v
    layer.out_projection.weights.value = w_out

    # The same mask is causal mask
    mask = -np.finfo(datatype).max * np.triu(np.ones((queries_len, queries_len), dtype=datatype), 1)
    M = torch.tensor(mask, dtype=torch.float32)

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

    w_qkv_grad = layer_.in_proj_weight.grad.detach().numpy().T.copy() # (n_embd, 3*n_embd)
    w_q_grad, w_k_grad, w_v_grad = [w.copy() for w in np.split(w_qkv_grad, 3, -1)] # 3 * (n_embd, n_embd)

    np.testing.assert_allclose(
        layer.q_projection.weights.value.grad.to_numpy(),
        w_q_grad,
        atol=1e-5,
        rtol=1e-5
    )
    np.testing.assert_allclose(
        layer.k_projection.weights.value.grad.to_numpy(),
        w_k_grad,
        atol=1e-5,
        rtol=1e-5
    )
    np.testing.assert_allclose(
        layer.v_projection.weights.value.grad.to_numpy(),
        w_v_grad,
        atol=1e-5,
        rtol=1e-5
    )

    np.save(os.path.join(test_dir, f'{test_str}_data.npy'), data)
    np.save(os.path.join(test_dir, f'{test_str}_w_q.npy'), w_q_)
    np.save(os.path.join(test_dir, f'{test_str}_w_k.npy'), w_k_)
    np.save(os.path.join(test_dir, f'{test_str}_w_v.npy'), w_v_)
    np.save(os.path.join(test_dir, f'{test_str}_w_out.npy'), w_out_)
    np.save(os.path.join(test_dir, f'{test_str}_result.npy'), result_.detach().numpy())
    np.save(os.path.join(test_dir, f'{test_str}_x_grad.npy'), X_.grad.detach().numpy())
    np.save(os.path.join(test_dir, f'{test_str}_w_q_grad.npy'), w_q_grad)
    np.save(os.path.join(test_dir, f'{test_str}_w_k_grad.npy'), w_k_grad)
    np.save(os.path.join(test_dir, f'{test_str}_w_v_grad.npy'), w_v_grad)
    np.save(os.path.join(test_dir, f'{test_str}_w_out_grad.npy'), layer_.out_proj.weight.grad.detach().numpy().T)


@pytest.mark.ref_teacher_a2_4
@pytest.mark.parametrize("batch_size",  [1, 128])
@pytest.mark.parametrize("queries_len", [32, 40])
@pytest.mark.parametrize("n_embd",      [64, 256])
@pytest.mark.parametrize("num_heads",   [1, 4, 8])
@pytest.mark.parametrize("p_dropout",   [0.0])
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_ref_teacher_multihead_attention(batch_size, queries_len, n_embd, num_heads, p_dropout, backend):
    np.random.seed(20)
    torch.manual_seed(20)

    test_dir = f'./tests/data_teacher/multihead_attention'
    test_str = '_'.join(map(str, (batch_size, queries_len, n_embd, num_heads)))
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    # data = np.random.randn(batch_size, queries_len, n_embd)
    data = np.random.rand(batch_size, queries_len, n_embd) # NOTE: rand works, not randn
    X    = minitorch.tensor_from_numpy(data, backend, True)
    X_   = torch.tensor(data, dtype=torch.float32, requires_grad=True)

    layer_ = torch.nn.MultiheadAttention(n_embd, num_heads, p_dropout, bias=False, batch_first=True, dtype=torch.float32)
    layer = minitorch.MultiHeadAttention(n_embd, num_heads, True, p_dropout, bias=False, backend=backend)
    
    # Set weights of minitorch layer to torch weights
    w_qkv = layer_.in_proj_weight.detach().numpy().T.copy() # (n_embd, 3*n_embd)
    w_q_, w_k_, w_v_ = [w.copy() for w in np.split(w_qkv, 3, -1)] # 3 * (n_embd, n_embd)
    w_out_ = layer_.out_proj.weight.detach().numpy().T.copy()

    w_q = minitorch.tensor_from_numpy((w_q_), backend=backend, requires_grad=True)
    w_k = minitorch.tensor_from_numpy((w_k_), backend=backend, requires_grad=True)
    w_v = minitorch.tensor_from_numpy((w_v_), backend=backend, requires_grad=True)
    w_out = minitorch.tensor_from_numpy((w_out_), backend=backend, requires_grad=True)

    layer.q_projection.weights.value = w_q
    layer.k_projection.weights.value = w_k
    layer.v_projection.weights.value = w_v
    layer.out_projection.weights.value = w_out

    # The same mask is causal mask
    mask = -np.finfo(datatype).max * np.triu(np.ones((queries_len, queries_len), dtype=datatype), 1)
    M = torch.tensor(mask, dtype=torch.float32)

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

    w_qkv_grad = layer_.in_proj_weight.grad.detach().numpy().T.copy() # (n_embd, 3*n_embd)
    w_q_grad, w_k_grad, w_v_grad = [w.copy() for w in np.split(w_qkv_grad, 3, -1)] # 3 * (n_embd, n_embd)

    np.testing.assert_allclose(
        layer.q_projection.weights.value.grad.to_numpy(),
        w_q_grad,
        atol=1e-5,
        rtol=1e-5
    )
    np.testing.assert_allclose(
        layer.k_projection.weights.value.grad.to_numpy(),
        w_k_grad,
        atol=1e-5,
        rtol=1e-5
    )
    np.testing.assert_allclose(
        layer.v_projection.weights.value.grad.to_numpy(),
        w_v_grad,
        atol=1e-5,
        rtol=1e-5
    )

    np.save(os.path.join(test_dir, f'{test_str}_data.npy'), data)
    np.save(os.path.join(test_dir, f'{test_str}_w_q.npy'), w_q_)
    np.save(os.path.join(test_dir, f'{test_str}_w_k.npy'), w_k_)
    np.save(os.path.join(test_dir, f'{test_str}_w_v.npy'), w_v_)
    np.save(os.path.join(test_dir, f'{test_str}_w_out.npy'), w_out_)
    np.save(os.path.join(test_dir, f'{test_str}_result.npy'), result_.detach().numpy())
    np.save(os.path.join(test_dir, f'{test_str}_x_grad.npy'), X_.grad.detach().numpy())
    np.save(os.path.join(test_dir, f'{test_str}_w_q_grad.npy'), w_q_grad)
    np.save(os.path.join(test_dir, f'{test_str}_w_k_grad.npy'), w_k_grad)
    np.save(os.path.join(test_dir, f'{test_str}_w_v_grad.npy'), w_v_grad)
    np.save(os.path.join(test_dir, f'{test_str}_w_out_grad.npy'), layer_.out_proj.weight.grad.detach().numpy().T)

################################ FFN ########################################

@pytest.mark.ref_student_a2_4
@pytest.mark.parametrize("batch_size", [1, 128])
@pytest.mark.parametrize("seq_len", [5, 40])
@pytest.mark.parametrize("n_embd",  [9, 256])
@pytest.mark.parametrize("dropout", [0.0])
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_ref_student_feedforward_layer(batch_size, seq_len, n_embd, dropout, backend):

    np.random.seed(11868)

    data = np.random.randn(batch_size, seq_len, n_embd).astype(datatype)

    layer = minitorch.FeedForward(n_embd=n_embd, p_dropout=dropout, bias=True, backend=backend)

    X = minitorch.tensor(data.tolist(), backend=backend, requires_grad=True)

    result = layer(X)

    assert result is not None
    assert not np.isnan(result.to_numpy()).any()

    result.sum().backward()

    assert X.grad is not None
    assert layer.linear_in.weights.value.grad is not None
    assert layer.linear_out.weights.value.grad is not None


@pytest.mark.ref_teacher_a2_4
@pytest.mark.parametrize("batch_size", [1, 128])
@pytest.mark.parametrize("seq_len", [5, 40])
@pytest.mark.parametrize("n_embd",  [9, 256])
@pytest.mark.parametrize("dropout", [0.0])
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_ref_teacher_feedforward_layer(batch_size, seq_len, n_embd, dropout, backend):

    np.random.seed(20)

    data = np.random.randn(batch_size, seq_len, n_embd).astype(datatype)

    layer = minitorch.FeedForward(n_embd=n_embd, p_dropout=dropout, bias=True, backend=backend)

    X = minitorch.tensor(data.tolist(), backend=backend, requires_grad=True)

    result = layer(X)

    assert result is not None
    assert not np.isnan(result.to_numpy()).any()

    result.sum().backward()

    assert X.grad is not None
    assert layer.linear_in.weights.value.grad is not None
    assert layer.linear_out.weights.value.grad is not None

################################ TRANSFORMER LAYER ########################################

@pytest.mark.ref_student_a2_4
@pytest.mark.parametrize("batch_size", [128])
@pytest.mark.parametrize("seq_len",   [40])
@pytest.mark.parametrize("n_embd",    [256])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("p_dropout", [0.0])
@pytest.mark.parametrize("ln_eps", [1e-5])
@pytest.mark.parametrize("bias", [False])
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_ref_student_transformer_layer_1(batch_size, seq_len, n_embd, num_heads, p_dropout, ln_eps, bias, backend):
    np.random.seed(10)
    torch.manual_seed(10)
    test_dir = f'./tests/data/transformer_layer_1'
    test_str = '_'.join(map(str, (batch_size, seq_len, n_embd, num_heads)))
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    data = np.random.rand(batch_size, seq_len, n_embd)
    X    = minitorch.tensor_from_numpy(data.copy(), backend, True)
    X_   = torch.tensor(data, dtype=torch.float32, requires_grad=True)

    layer_ = torch.nn.TransformerEncoderLayer(
        d_model=n_embd, nhead=num_heads, dim_feedforward=256, dropout=p_dropout,
        activation=lambda x: torch.nn.functional.gelu(x, approximate='tanh'),
        layer_norm_eps=ln_eps, batch_first=True, norm_first=True, bias=bias, dtype=torch.float32
    )

    layer = minitorch.TransformerLayer(
        n_embd=n_embd, n_head=num_heads, p_dropout=p_dropout, ln_eps=ln_eps, 
        bias=bias, backend=backend
    )
    

    # FFN Weights
    w_ffn_in = layer_.linear1.weight.detach().numpy().T.copy()
    w_ffn_out = layer_.linear2.weight.detach().numpy().T.copy()
    
    # Transformer Weights
    w_qkv = layer_.self_attn.in_proj_weight.detach().numpy().T.copy() # (n_embd, 3*n_embd)
    w_q_, w_k_, w_v_ = [w.copy() for w in np.split(w_qkv, 3, -1)] # 3 * (n_embd, n_embd)
    w_out_ = layer_.self_attn.out_proj.weight.detach().numpy().T.copy()

    # Set weights (LN will be 0s and 1s so it's ok to ignore for now)
    layer.attention.q_projection.weights.value = minitorch.tensor_from_numpy(w_q_, backend=backend, requires_grad=True)
    layer.attention.k_projection.weights.value = minitorch.tensor_from_numpy(w_k_, backend=backend, requires_grad=True)
    layer.attention.v_projection.weights.value = minitorch.tensor_from_numpy(w_v_, backend=backend, requires_grad=True)
    layer.attention.out_projection.weights.value = minitorch.tensor_from_numpy(w_out_, backend=backend, requires_grad=True)
    layer.ff.linear_in.weights.value = minitorch.tensor_from_numpy(w_ffn_in, backend=backend, requires_grad=True)
    layer.ff.linear_out.weights.value = minitorch.tensor_from_numpy(w_ffn_out, backend=backend, requires_grad=True)

    # Mask for Torch
    mask = -np.finfo(datatype).max * np.triu(np.ones((seq_len, seq_len), dtype=datatype), 1)
    M = torch.tensor(mask, dtype=torch.float32)

    result = layer(X)
    result_ = layer_(X_, M)

    assert result is not None
    np.testing.assert_allclose(
        result.to_numpy(),
        result_.detach().numpy(),
        atol=1e-5,
        rtol=1e-5
    )

    result.sum().backward()
    result_.sum().backward()

    np.testing.assert_allclose(
        X.grad.to_numpy(), 
        X_.grad.detach().numpy(), 
        atol=1e-5,
        rtol=1e-5
    )

    np.save(os.path.join(test_dir, f'{test_str}_data.npy'), data)
    np.save(os.path.join(test_dir, f'{test_str}_w_ffn_in.npy'), w_ffn_in)
    np.save(os.path.join(test_dir, f'{test_str}_w_ffn_out.npy'), w_ffn_out)
    np.save(os.path.join(test_dir, f'{test_str}_w_q.npy'), w_q_)
    np.save(os.path.join(test_dir, f'{test_str}_w_k.npy'), w_k_)
    np.save(os.path.join(test_dir, f'{test_str}_w_v.npy'), w_v_)
    np.save(os.path.join(test_dir, f'{test_str}_w_out.npy'), w_out_)
    np.save(os.path.join(test_dir, f'{test_str}_result.npy'), result_.detach().numpy())
    np.save(os.path.join(test_dir, f'{test_str}_x_grad.npy'), X_.grad.detach().numpy())


@pytest.mark.ref_teacher_a2_4
@pytest.mark.parametrize("batch_size", [128])
@pytest.mark.parametrize("seq_len",   [40])
@pytest.mark.parametrize("n_embd",    [256])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("p_dropout", [0.0])
@pytest.mark.parametrize("ln_eps", [1e-5])
@pytest.mark.parametrize("bias", [False])
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_ref_teacher_transformer_layer_1(batch_size, seq_len, n_embd, num_heads, p_dropout, ln_eps, bias, backend):
    np.random.seed(20)
    torch.manual_seed(20)
    test_dir = f'./tests/data_teacher/transformer_layer_1'
    test_str = '_'.join(map(str, (batch_size, seq_len, n_embd, num_heads)))
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    data = np.random.rand(batch_size, seq_len, n_embd)
    X    = minitorch.tensor_from_numpy(data.copy(), backend, True)
    X_   = torch.tensor(data, dtype=torch.float32, requires_grad=True)

    layer_ = torch.nn.TransformerEncoderLayer(
        d_model=n_embd, nhead=num_heads, dim_feedforward=256, dropout=p_dropout,
        activation=lambda x: torch.nn.functional.gelu(x, approximate='tanh'),
        layer_norm_eps=ln_eps, batch_first=True, norm_first=True, bias=bias, dtype=torch.float32
    )

    layer = minitorch.TransformerLayer(
        n_embd=n_embd, n_head=num_heads, p_dropout=p_dropout, ln_eps=ln_eps, 
        bias=bias, backend=backend
    )
    

    # FFN Weights
    w_ffn_in = layer_.linear1.weight.detach().numpy().T.copy()
    w_ffn_out = layer_.linear2.weight.detach().numpy().T.copy()
    
    # Transformer Weights
    w_qkv = layer_.self_attn.in_proj_weight.detach().numpy().T.copy() # (n_embd, 3*n_embd)
    w_q_, w_k_, w_v_ = [w.copy() for w in np.split(w_qkv, 3, -1)] # 3 * (n_embd, n_embd)
    w_out_ = layer_.self_attn.out_proj.weight.detach().numpy().T.copy()

    # Set weights (LN will be 0s and 1s so it's ok to ignore for now)
    layer.attention.q_projection.weights.value = minitorch.tensor_from_numpy(w_q_, backend=backend, requires_grad=True)
    layer.attention.k_projection.weights.value = minitorch.tensor_from_numpy(w_k_, backend=backend, requires_grad=True)
    layer.attention.v_projection.weights.value = minitorch.tensor_from_numpy(w_v_, backend=backend, requires_grad=True)
    layer.attention.out_projection.weights.value = minitorch.tensor_from_numpy(w_out_, backend=backend, requires_grad=True)
    layer.ff.linear_in.weights.value = minitorch.tensor_from_numpy(w_ffn_in, backend=backend, requires_grad=True)
    layer.ff.linear_out.weights.value = minitorch.tensor_from_numpy(w_ffn_out, backend=backend, requires_grad=True)

    # Mask for Torch
    mask = -np.finfo(datatype).max * np.triu(np.ones((seq_len, seq_len), dtype=datatype), 1)
    M = torch.tensor(mask, dtype=torch.float32)

    result = layer(X)
    result_ = layer_(X_, M)

    assert result is not None
    np.testing.assert_allclose(
        result.to_numpy(),
        result_.detach().numpy(),
        atol=1e-5,
        rtol=1e-5
    )

    result.sum().backward()
    result_.sum().backward()

    np.testing.assert_allclose(
        X.grad.to_numpy(), 
        X_.grad.detach().numpy(), 
        atol=1e-5,
        rtol=1e-5
    )

    np.save(os.path.join(test_dir, f'{test_str}_data.npy'), data)
    np.save(os.path.join(test_dir, f'{test_str}_w_ffn_in.npy'), w_ffn_in)
    np.save(os.path.join(test_dir, f'{test_str}_w_ffn_out.npy'), w_ffn_out)
    np.save(os.path.join(test_dir, f'{test_str}_w_q.npy'), w_q_)
    np.save(os.path.join(test_dir, f'{test_str}_w_k.npy'), w_k_)
    np.save(os.path.join(test_dir, f'{test_str}_w_v.npy'), w_v_)
    np.save(os.path.join(test_dir, f'{test_str}_w_out.npy'), w_out_)
    np.save(os.path.join(test_dir, f'{test_str}_result.npy'), result_.detach().numpy())
    np.save(os.path.join(test_dir, f'{test_str}_x_grad.npy'), X_.grad.detach().numpy())


@pytest.mark.ref_student_a2_4
@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("seq_len",   [4, 32])
@pytest.mark.parametrize("n_embd",    [16, 32])
@pytest.mark.parametrize("num_heads", [1, 8])
@pytest.mark.parametrize("p_dropout", [0.0])
@pytest.mark.parametrize("ln_eps", [1e-5])
@pytest.mark.parametrize("bias", [False])
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_ref_student_transformer_layer_2(batch_size, seq_len, n_embd, num_heads, p_dropout, ln_eps, bias, backend):
    np.random.seed(10)
    torch.manual_seed(10)
    test_dir = f'./tests/data/transformer_layer_2'
    test_str = '_'.join(map(str, (batch_size, seq_len, n_embd, num_heads)))
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    data = np.random.rand(batch_size, seq_len, n_embd)
    X    = minitorch.tensor_from_numpy(data.copy(), backend, True)
    X_   = torch.tensor(data, dtype=torch.float32, requires_grad=True)

    layer_ = torch.nn.TransformerEncoderLayer(
        d_model=n_embd, nhead=num_heads, dim_feedforward=256, dropout=p_dropout,
        activation=lambda x: torch.nn.functional.gelu(x, approximate='tanh'),
        layer_norm_eps=ln_eps, batch_first=True, norm_first=True, bias=bias, dtype=torch.float32
    )

    layer = minitorch.TransformerLayer(
        n_embd=n_embd, n_head=num_heads, p_dropout=p_dropout, ln_eps=ln_eps, 
        bias=bias, backend=backend
    )
    

    # FFN Weights
    w_ffn_in = layer_.linear1.weight.detach().numpy().T.copy()
    w_ffn_out = layer_.linear2.weight.detach().numpy().T.copy()
    
    # Transformer Weights
    w_qkv = layer_.self_attn.in_proj_weight.detach().numpy().T.copy() # (n_embd, 3*n_embd)
    w_q_, w_k_, w_v_ = [w.copy() for w in np.split(w_qkv, 3, -1)] # 3 * (n_embd, n_embd)
    w_out_ = layer_.self_attn.out_proj.weight.detach().numpy().T.copy()

    # Set weights (LN will be 0s and 1s so it's ok to ignore for now)
    layer.attention.q_projection.weights.value = minitorch.tensor_from_numpy(w_q_, backend=backend, requires_grad=True)
    layer.attention.k_projection.weights.value = minitorch.tensor_from_numpy(w_k_, backend=backend, requires_grad=True)
    layer.attention.v_projection.weights.value = minitorch.tensor_from_numpy(w_v_, backend=backend, requires_grad=True)
    layer.attention.out_projection.weights.value = minitorch.tensor_from_numpy(w_out_, backend=backend, requires_grad=True)
    layer.ff.linear_in.weights.value = minitorch.tensor_from_numpy(w_ffn_in, backend=backend, requires_grad=True)
    layer.ff.linear_out.weights.value = minitorch.tensor_from_numpy(w_ffn_out, backend=backend, requires_grad=True)

    # Mask for Torch
    mask = -np.finfo(datatype).max * np.triu(np.ones((seq_len, seq_len), dtype=datatype), 1)
    M = torch.tensor(mask, dtype=torch.float32)

    result = layer(X)
    result_ = layer_(X_, M)

    assert result is not None
    np.testing.assert_allclose(
        result.to_numpy(),
        result_.detach().numpy(),
        atol=1e-5,
        rtol=1e-5
    )

    result.sum().backward()
    result_.sum().backward()

    np.testing.assert_allclose(
        X.grad.to_numpy(), 
        X_.grad.detach().numpy(), 
        atol=1e-5,
        rtol=1e-5
    )

    # Let's check some weights
    np.testing.assert_allclose(
        layer.attention.out_projection.weights.value.grad.to_numpy(), 
        layer_.self_attn.out_proj.weight.grad.detach().numpy().T,
        atol=1e-5, 
        rtol=1e-5
    )
    np.testing.assert_allclose(
        layer.ff.linear_out.weights.value.grad.to_numpy(),
        layer_.linear2.weight.grad.detach().numpy().T, 
        atol=1e-5,
        rtol=1e-5
    )
    np.testing.assert_allclose(
        layer.ff.linear_in.weights.value.grad.to_numpy(),
        layer_.linear1.weight.grad.detach().numpy().T.astype(np.float32), 
        atol=1e-5,
        rtol=1e-5
    )

    w_qkv_grad = layer_.self_attn.in_proj_weight.grad.detach().numpy().T.copy() # (n_embd, 3*n_embd)
    w_q_grad, w_k_grad, w_v_grad = [w.copy() for w in np.split(w_qkv_grad, 3, -1)] # 3 * (n_embd, n_embd)

    np.testing.assert_allclose(
        layer.attention.q_projection.weights.value.grad.to_numpy(),
        w_q_grad,
        atol=1e-5,
        rtol=1e-5
    )
    np.testing.assert_allclose(
        layer.attention.k_projection.weights.value.grad.to_numpy(),
        w_k_grad,
        atol=1e-5,
        rtol=1e-5
    )
    np.testing.assert_allclose(
        layer.attention.v_projection.weights.value.grad.to_numpy(),
        w_v_grad,
        atol=1e-5,
        rtol=1e-5
    )

    np.save(os.path.join(test_dir, f'{test_str}_data.npy'), data)
    np.save(os.path.join(test_dir, f'{test_str}_w_ffn_in.npy'), w_ffn_in)
    np.save(os.path.join(test_dir, f'{test_str}_w_ffn_out.npy'), w_ffn_out)
    np.save(os.path.join(test_dir, f'{test_str}_w_q.npy'), w_q_)
    np.save(os.path.join(test_dir, f'{test_str}_w_k.npy'), w_k_)
    np.save(os.path.join(test_dir, f'{test_str}_w_v.npy'), w_v_)
    np.save(os.path.join(test_dir, f'{test_str}_w_out.npy'), w_out_)
    np.save(os.path.join(test_dir, f'{test_str}_result.npy'), result_.detach().numpy())
    np.save(os.path.join(test_dir, f'{test_str}_x_grad.npy'), X_.grad.detach().numpy())
    np.save(os.path.join(test_dir, f'{test_str}_w_ffn_in_grad.npy'), layer_.linear1.weight.grad.detach().numpy().T.astype(np.float32))
    np.save(os.path.join(test_dir, f'{test_str}_w_ffn_out_grad.npy'), layer_.linear2.weight.grad.detach().numpy().T)
    np.save(os.path.join(test_dir, f'{test_str}_w_out_grad.npy'), layer_.self_attn.out_proj.weight.grad.detach().numpy().T)
    np.save(os.path.join(test_dir, f'{test_str}_w_q_grad.npy'), w_q_grad)
    np.save(os.path.join(test_dir, f'{test_str}_w_k_grad.npy'), w_k_grad)
    np.save(os.path.join(test_dir, f'{test_str}_w_v_grad.npy'), w_v_grad)


@pytest.mark.ref_teacher_a2_4
@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("seq_len",   [4, 32])
@pytest.mark.parametrize("n_embd",    [16, 32])
@pytest.mark.parametrize("num_heads", [1, 8])
@pytest.mark.parametrize("p_dropout", [0.0])
@pytest.mark.parametrize("ln_eps", [1e-5])
@pytest.mark.parametrize("bias", [False])
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_ref_teacher_transformer_layer_2(batch_size, seq_len, n_embd, num_heads, p_dropout, ln_eps, bias, backend):
    np.random.seed(20)
    torch.manual_seed(20)
    test_dir = f'./tests/data_teacher/transformer_layer_2'
    test_str = '_'.join(map(str, (batch_size, seq_len, n_embd, num_heads)))
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    data = np.random.rand(batch_size, seq_len, n_embd)
    X    = minitorch.tensor_from_numpy(data.copy(), backend, True)
    X_   = torch.tensor(data, dtype=torch.float32, requires_grad=True)

    layer_ = torch.nn.TransformerEncoderLayer(
        d_model=n_embd, nhead=num_heads, dim_feedforward=256, dropout=p_dropout,
        activation=lambda x: torch.nn.functional.gelu(x, approximate='tanh'),
        layer_norm_eps=ln_eps, batch_first=True, norm_first=True, bias=bias, dtype=torch.float32
    )

    layer = minitorch.TransformerLayer(
        n_embd=n_embd, n_head=num_heads, p_dropout=p_dropout, ln_eps=ln_eps, 
        bias=bias, backend=backend
    )
    

    # FFN Weights
    w_ffn_in = layer_.linear1.weight.detach().numpy().T.copy()
    w_ffn_out = layer_.linear2.weight.detach().numpy().T.copy()
    
    # Transformer Weights
    w_qkv = layer_.self_attn.in_proj_weight.detach().numpy().T.copy() # (n_embd, 3*n_embd)
    w_q_, w_k_, w_v_ = [w.copy() for w in np.split(w_qkv, 3, -1)] # 3 * (n_embd, n_embd)
    w_out_ = layer_.self_attn.out_proj.weight.detach().numpy().T.copy()

    # Set weights (LN will be 0s and 1s so it's ok to ignore for now)
    layer.attention.q_projection.weights.value = minitorch.tensor_from_numpy(w_q_, backend=backend, requires_grad=True)
    layer.attention.k_projection.weights.value = minitorch.tensor_from_numpy(w_k_, backend=backend, requires_grad=True)
    layer.attention.v_projection.weights.value = minitorch.tensor_from_numpy(w_v_, backend=backend, requires_grad=True)
    layer.attention.out_projection.weights.value = minitorch.tensor_from_numpy(w_out_, backend=backend, requires_grad=True)
    layer.ff.linear_in.weights.value = minitorch.tensor_from_numpy(w_ffn_in, backend=backend, requires_grad=True)
    layer.ff.linear_out.weights.value = minitorch.tensor_from_numpy(w_ffn_out, backend=backend, requires_grad=True)

    # Mask for Torch
    mask = -np.finfo(datatype).max * np.triu(np.ones((seq_len, seq_len), dtype=datatype), 1)
    M = torch.tensor(mask, dtype=torch.float32)

    result = layer(X)
    result_ = layer_(X_, M)

    assert result is not None
    np.testing.assert_allclose(
        result.to_numpy(),
        result_.detach().numpy(),
        atol=1e-5,
        rtol=1e-5
    )

    result.sum().backward()
    result_.sum().backward()

    np.testing.assert_allclose(
        X.grad.to_numpy(), 
        X_.grad.detach().numpy(), 
        atol=1e-5,
        rtol=1e-5
    )

    # Let's check some weights
    np.testing.assert_allclose(
        layer.attention.out_projection.weights.value.grad.to_numpy(), 
        layer_.self_attn.out_proj.weight.grad.detach().numpy().T,
        atol=1e-5, 
        rtol=1e-5
    )
    np.testing.assert_allclose(
        layer.ff.linear_out.weights.value.grad.to_numpy(),
        layer_.linear2.weight.grad.detach().numpy().T, 
        atol=1e-5,
        rtol=1e-5
    )
    np.testing.assert_allclose(
        layer.ff.linear_in.weights.value.grad.to_numpy(),
        layer_.linear1.weight.grad.detach().numpy().T.astype(np.float32), 
        atol=1e-5,
        rtol=1e-5
    )

    w_qkv_grad = layer_.self_attn.in_proj_weight.grad.detach().numpy().T.copy() # (n_embd, 3*n_embd)
    w_q_grad, w_k_grad, w_v_grad = [w.copy() for w in np.split(w_qkv_grad, 3, -1)] # 3 * (n_embd, n_embd)

    np.testing.assert_allclose(
        layer.attention.q_projection.weights.value.grad.to_numpy(),
        w_q_grad,
        atol=1e-5,
        rtol=1e-5
    )
    np.testing.assert_allclose(
        layer.attention.k_projection.weights.value.grad.to_numpy(),
        w_k_grad,
        atol=1e-5,
        rtol=1e-5
    )
    np.testing.assert_allclose(
        layer.attention.v_projection.weights.value.grad.to_numpy(),
        w_v_grad,
        atol=1e-5,
        rtol=1e-5
    )

    np.save(os.path.join(test_dir, f'{test_str}_data.npy'), data)
    np.save(os.path.join(test_dir, f'{test_str}_w_ffn_in.npy'), w_ffn_in)
    np.save(os.path.join(test_dir, f'{test_str}_w_ffn_out.npy'), w_ffn_out)
    np.save(os.path.join(test_dir, f'{test_str}_w_q.npy'), w_q_)
    np.save(os.path.join(test_dir, f'{test_str}_w_k.npy'), w_k_)
    np.save(os.path.join(test_dir, f'{test_str}_w_v.npy'), w_v_)
    np.save(os.path.join(test_dir, f'{test_str}_w_out.npy'), w_out_)
    np.save(os.path.join(test_dir, f'{test_str}_result.npy'), result_.detach().numpy())
    np.save(os.path.join(test_dir, f'{test_str}_x_grad.npy'), X_.grad.detach().numpy())
    np.save(os.path.join(test_dir, f'{test_str}_w_ffn_in_grad.npy'), layer_.linear1.weight.grad.detach().numpy().T.astype(np.float32))
    np.save(os.path.join(test_dir, f'{test_str}_w_ffn_out_grad.npy'), layer_.linear2.weight.grad.detach().numpy().T)
    np.save(os.path.join(test_dir, f'{test_str}_w_out_grad.npy'), layer_.self_attn.out_proj.weight.grad.detach().numpy().T)
    np.save(os.path.join(test_dir, f'{test_str}_w_q_grad.npy'), w_q_grad)
    np.save(os.path.join(test_dir, f'{test_str}_w_k_grad.npy'), w_k_grad)
    np.save(os.path.join(test_dir, f'{test_str}_w_v_grad.npy'), w_v_grad)



@pytest.mark.ref_student_a2_4
@pytest.mark.parametrize("batch_size", [128])
@pytest.mark.parametrize("seq_len", [40])
@pytest.mark.parametrize("n_vocab", [1000])
@pytest.mark.parametrize("n_embd",  [256])
@pytest.mark.parametrize("n_head",  [8])
@pytest.mark.parametrize("n_positions", [40])
@pytest.mark.parametrize("dropout", [0.0])
@pytest.mark.parametrize("ln_eps", [1e-5])
@pytest.mark.parametrize("bias", [True])
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_ref_student_decoder_lm(batch_size, seq_len, n_vocab, n_embd, n_head, n_positions, dropout, ln_eps, bias, backend):

    np.random.seed(10)
    x = np.random.randint(low=0, high=n_vocab, size=(batch_size, seq_len))

    layer = minitorch.DecoderLM(
        n_vocab=n_vocab, n_embd=n_embd, n_head=n_head, n_positions=n_positions, 
        p_dropout=dropout, ln_eps=ln_eps, bias=bias, backend=backend)

    result = layer(minitorch.tensor(x.tolist(), backend=backend, requires_grad=True))

    assert result is not None
    assert not np.isnan(result.to_numpy()).any()
    assert result.shape == (batch_size, seq_len, n_vocab)

    result.sum().backward()

    assert layer.position_embeddings.weights.value.grad is not None
    assert layer.token_embeddings.weights.value.grad is not None
    assert not np.isnan(layer.position_embeddings.weights.value.grad.to_numpy()).any()
    assert not np.isnan(layer.token_embeddings.weights.value.grad.to_numpy()).any()


@pytest.mark.ref_teacher_a2_4
@pytest.mark.parametrize("batch_size", [128])
@pytest.mark.parametrize("seq_len", [40])
@pytest.mark.parametrize("n_vocab", [1000])
@pytest.mark.parametrize("n_embd",  [256])
@pytest.mark.parametrize("n_head",  [8])
@pytest.mark.parametrize("n_positions", [40])
@pytest.mark.parametrize("dropout", [0.0])
@pytest.mark.parametrize("ln_eps", [1e-5])
@pytest.mark.parametrize("bias", [True])
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_ref_teacher_decoder_lm(batch_size, seq_len, n_vocab, n_embd, n_head, n_positions, dropout, ln_eps, bias, backend):

    np.random.seed(20)
    x = np.random.randint(low=0, high=n_vocab, size=(batch_size, seq_len))

    layer = minitorch.DecoderLM(
        n_vocab=n_vocab, n_embd=n_embd, n_head=n_head, n_positions=n_positions, 
        p_dropout=dropout, ln_eps=ln_eps, bias=bias, backend=backend)

    result = layer(minitorch.tensor(x.tolist(), backend=backend, requires_grad=True))

    assert result is not None
    assert not np.isnan(result.to_numpy()).any()
    assert result.shape == (batch_size, seq_len, n_vocab)

    result.sum().backward()

    assert layer.position_embeddings.weights.value.grad is not None
    assert layer.token_embeddings.weights.value.grad is not None
    assert not np.isnan(layer.position_embeddings.weights.value.grad.to_numpy()).any()
    assert not np.isnan(layer.token_embeddings.weights.value.grad.to_numpy()).any()



"""
TESTING NOTES:
# batch_size = 64, queries_len = 256, n_embd = 64, num_heads = 1, p_dropout = 0.0, backend = <minitorch.tensor_ops.TensorBackend object at 0x7faacd7a8d00>
    tests/test_modules_transformer.py::test_multihead_attention[CudaKernelOps-0.0-1-64-2-1] PASSED                                                                                                                                                 [  6%]
tests/test_modules_transformer.py::test_multihead_attention[CudaKernelOps-0.0-1-64-2-64] PASSED                                                                                                                                                [ 12%]
tests/test_modules_transformer.py::test_multihead_attention[CudaKernelOps-0.0-1-64-256-1] PASSED                                                                                                                                               [ 18%]
tests/test_modules_transformer.py::test_multihead_attention[CudaKernelOps-0.0-1-64-256-64] FAILED                                                                                                                                              [ 25%]
tests/test_modules_transformer.py::test_multihead_attention[CudaKernelOps-0.0-1-256-2-1] PASSED                                                                                                                                                [ 31%]
tests/test_modules_transformer.py::test_multihead_attention[CudaKernelOps-0.0-1-256-2-64] PASSED                                                                                                                                               [ 37%]
tests/test_modules_transformer.py::test_multihead_attention[CudaKernelOps-0.0-1-256-256-1] PASSED                                                                                                                                              [ 43%]
tests/test_modules_transformer.py::test_multihead_attention[CudaKernelOps-0.0-1-256-256-64] FAILED                                                                                                                                             [ 50%]
tests/test_modules_transformer.py::test_multihead_attention[CudaKernelOps-0.0-4-64-2-1] PASSED                                                                                                                                                 [ 56%]
tests/test_modules_transformer.py::test_multihead_attention[CudaKernelOps-0.0-4-64-2-64] PASSED                                                                                                                                                [ 62%]
tests/test_modules_transformer.py::test_multihead_attention[CudaKernelOps-0.0-4-64-256-1] PASSED                                                                                                                                               [ 68%]
tests/test_modules_transformer.py::test_multihead_attention[CudaKernelOps-0.0-4-64-256-64] PASSED                                                                                                                                              [ 75%]
tests/test_modules_transformer.py::test_multihead_attention[CudaKernelOps-0.0-4-256-2-1] PASSED                                                                                                                                                [ 81%]
tests/test_modules_transformer.py::test_multihead_attention[CudaKernelOps-0.0-4-256-2-64] PASSED                                                                                                                                               [ 87%]
tests/test_modules_transformer.py::test_multihead_attention[CudaKernelOps-0.0-4-256-256-1] PASSED                                                                                                                                              [ 93%]
tests/test_modules_transformer.py::test_multihead_attention[CudaKernelOps-0.0-4-256-256-64] FAILED   

"""
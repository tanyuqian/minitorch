from typing import Tuple
import numba
from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .cuda_ops import CudaOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor, tensor_from_numpy
import numpy as np
import math

def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """
    Reshape an image tensor for 2D pooling

    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.
    """

    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # ASSIGN4.3
    new_width = width // kw
    new_height = height // kh

    x = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)
    x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
    x = x.view(batch, channel, new_height, new_width, kh * kw)
    return x, new_height, new_width
    # END ASSIGN4.3


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """
    Tiled average pooling 2D

    Args:
        input : batch x channel x height x width
        kernel : height x width of pooling

    Returns:
        Pooled tensor
    """
    batch, channel, height, width = input.shape
    # ASSIGN4.3
    x, new_height, new_width = tile(input, kernel)
    return x.mean(dim=4).view(batch, channel, new_height, new_width)
    # END ASSIGN4.3

if numba.cuda.is_available():
    # max_reduce = CudaOps.reduce(operators.max, -1e9)
    from minitorch.cuda_kernel_ops import CudaKernelOps
    max_reduce = CudaKernelOps.reduce(operators.max, -1e9)
else:
    max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """
    Compute the argmax as a 1-hot tensor.

    Args:
        input : input tensor
        dim : dimension to apply argmax


    Returns:
        :class:`Tensor` : tensor with 1 on highest cell in dim, 0 otherwise

    """
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        "Forward of max should be max reduction"
        # ASSIGN4.4
        out = max_reduce(input, int(dim.item()))
        ctx.save_for_backward(input, out)
        return out
        # END ASSIGN4.4

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        "Backward of max should be argmax (see above)"
        # ASSIGN4.4
        input, out = ctx.saved_values
        return (out == input) * grad_output, 0.0
        # END ASSIGN4.4

        
def max(input: Tensor, dim: int) -> Tensor:
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    r"""
    Compute the softmax as a tensor.



    $z_i = \frac{e^{x_i}}{\sum_i e^{x_i}}$

    Args:
        input : input tensor
        dim : dimension to apply softmax

    Returns:
        softmax tensor
    """
    # ASSIGN4.4
    # e = (input - max(input, dim)).exp()
    e = (input).exp()
    partition = e.sum(dim=dim)
    return e / partition
    # END ASSIGN4.4


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    r"""
    Compute the log of the softmax as a tensor.

    $z_i = x_i - \log \sum_i e^{x_i}$

    See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations

    Args:
        input : input tensor
        dim : dimension to apply log-softmax

    Returns:
         log of softmax tensor
    """
    # ASSIGN4.4
    e = input
    mx = Max.apply(e, tensor([dim]))
    lse = (e - mx).exp().sum(dim=dim).log() + mx
    return e - lse
    # END ASSIGN4.4


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """
    Tiled max pooling 2D

    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
        Tensor : pooled tensor
    """
    batch, channel, height, width = input.shape
    # ASSIGN4.4
    x, new_height, new_width = tile(input, kernel)
    return max(x, 4).view(batch, channel, new_height, new_width)
    # END ASSIGN4.4


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """
    Dropout positions based on random noise.

    Args:
        input : input tensor
        rate : probability [0, 1) of dropping out each position
        ignore : skip dropout, i.e. do nothing at all

    Returns:
        tensor with random positions dropped out
    """
    # ASSIGN4.4
    if ignore:
        return input
    r = rand(input.shape, backend=input.backend)
    drop = rate < r
    return input * drop
    # END ASSIGN4.4

def GELU(input: Tensor) -> Tensor: 
    """
    https://github.com/karpathy/minGPT/blob/master/mingpt/model.py

    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    Note: This is the tanh approximation of GELU. 
    """
    return 0.5 * input * (1 + (np.sqrt(2 / math.pi) * (input + 0.044715 * (input ** 3))).tanh())

def layer_norm(input: Tensor, eps: float = 1e-5) -> Tensor:
    # ASSIGN4.4
    
    # Calculate mean and variance along the last axis (features)
    batch, channel, height, width = input.shape
    
    mean = input.mean(dim=4).view(batch, channel, height, width)
    variance = input.var(dim=4).view(batch, channel, height, width)
    
    input_normalized = (input - mean) / (variance + eps)
    return input_normalized

    # END ASSIGN4.4

def logsumexp(input: Tensor, dim: int) -> Tensor:
    """
    Calculates logsumexp with logsumexp trick
    """  
    input_max = max(input, dim=dim)
    out = ((input - input_max).exp().sum(dim).log()) + input_max

    return out


def softmax_loss(logits: Tensor, target: Tensor) -> Tensor:
    """
    Softmax + Cross Entropy Loss

    Args: 
        logits : (minibatch, C) Tensor of logits
        target : (minibatch, ) Tensor of true labels 

    Returns: 
        loss : (minibatch, )
    """
    batch_size, classes = logits.shape

    def one_hot(x):
        # Constructs np.eye(num_embeddings)[x] -> for each elem of x is a one hot
        # return tensor(
        #     np.eye(logits.shape[1])[x.to_numpy().astype(int)].tolist(), 
        #     backend=logits.backend
        # )
        return tensor_from_numpy(
            np.eye(logits.shape[1])[x.to_numpy().astype(int)], 
            backend=logits.backend
        )

    target_one_hot = one_hot(target)
    result = logsumexp(logits, dim=1) - (logits * target_one_hot).sum(1) # sum is to reduce
    # Flatten to be the same as torch.cross_entropy with reduction=None
    return result.view(batch_size, )
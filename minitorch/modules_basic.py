"""
For additional transformer related

Sequential
Embedding

"""
import numpy as np

from .module import Module, Parameter
from .tensor_functions import (zeros, ones, rand, tensor, tensor_from_numpy)
from .tensor_ops import TensorBackend
from .tensor import Tensor

from typing import Any, Dict, Optional, Sequence, Tuple


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, backend: TensorBackend):
        super().__init__()
        self.backend = backend
        self.num_embeddings = num_embeddings # Vocab size
        self.embedding_dim  = embedding_dim  # Embedding Dimension
        # Follow torch with initialization from standard normal bc. randn defaults to std. norm
        self.weights = Parameter(
            tensor_from_numpy(np.random.randn(num_embeddings, embedding_dim), backend=backend, requires_grad=True)
        )
    
    def one_hot(self, x):
        # Constructs np.eye(num_embeddings)[x] -> for each elem of x is a one hot
        return tensor_from_numpy(
            np.eye(self.num_embeddings)[x.to_numpy().astype(int)], 
            backend=self.backend
        )
    
    def forward(self, x):
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (bs, seq_len)

        Output:
        output of shape (bs, seq_len, embedding_dim)
        """
        bs, seq_len = x.shape
        x = x.view(seq_len * bs,)
        one_hot_x = self.one_hot(x)
        # (seq_len * bs, num_embed) -> (seq_len * bs, embedding_dim)
        out = one_hot_x @ self.weights.value.view(self.num_embeddings, self.embedding_dim)
        out = out.view(bs, seq_len, self.embedding_dim)

        return out

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.children = (modules)
    
    def forward(self, x):
        for module in self.children:
          x = module.forward(x)
        return x
    
    
class Dropout(Module):
    def __init__(self, p_dropout: float):
        super().__init__()
        self.p_dropout = p_dropout

    def forward(self, x: Tensor) -> Tensor: 
        if self.p_dropout == 0:
            return x
        elif self.training:
            mask = tensor_from_numpy(np.random.binomial(1, p=1-self.p_dropout, size=x.shape), backend=x.backend, requires_grad=False)
            return (x * mask) / (1 - self.p_dropout)
        else:
            return x


class Linear(Module):
    def __init__(self, in_size, out_size, bias, backend):
        super().__init__()
        self.weights = Parameter(rand((in_size, out_size), backend=backend, requires_grad=True))
        self.bias    = Parameter(rand((out_size,), backend=backend, requires_grad=True)) if bias else None
        self.out_size = out_size

    def forward(self, x):
        batch, in_size = x.shape

        result = (
            x.view(batch, in_size) @ self.weights.value.view(in_size, self.out_size)
        ).view(batch, self.out_size)        
        if self.bias is not None:
            result += self.bias.value

        return result


class LayerNorm1d(Module):
    def __init__(self, dim, eps, backend):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weights = Parameter(ones((self.dim, ), backend=backend))
        self.bias   = Parameter(zeros((self.dim, ), backend=backend))

        self.weights.value.requires_grad_(True)
        self.bias.value.requires_grad_(True)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:

        ### BEGIN YOUR SOLUTION
        batch, dim = x.shape
    
        x = x.contiguous()
        mean = x.mean(dim=1).view(batch, 1)
        variance = x.var(dim=1).view(batch, 1)
        # assert(False)
        x = (x - mean) / ((variance + self.eps) ** 0.5)
        x = (self.weights.value.view(1, self.dim) * x + 
                  self.bias.value.view(1, self.dim))

        return x
        ### END YOUR SOLUTION
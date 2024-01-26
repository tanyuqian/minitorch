import numpy as np
from .tensor import tensor, tensor_from_numpy
from .module import Module, Parameter
from .modules_basic import (
    Embedding,
    Sequential,
    Dropout,
    LayerNorm1d,
    Linear
)
from .tensor_ops import TensorBackend
from .nn import (
    max,
    softmax,
    dropout,
    GELU,
)
from typing import Any, Dict, Optional, Sequence, Tuple

datatype = np.float32


class MultiHeadAttention(Module):
    def __init__(self, n_embd: int, n_head: int, causal: bool=True, p_dropout: float=0.1, bias: bool=True, backend: TensorBackend=None):
        super().__init__()
        self.backend   = backend
        self.n_embd    = n_embd 
        self.n_head    = n_head
        self.causal    = causal
        self.attn_hidden_dim = n_embd // n_head

        self.q_projection = Linear(in_size=n_embd, out_size=n_embd, bias=bias, backend=backend)
        self.k_projection = Linear(in_size=n_embd, out_size=n_embd, bias=bias, backend=backend)
        self.v_projection = Linear(in_size=n_embd, out_size=n_embd, bias=bias, backend=backend)
        self.out_projection = Linear(in_size=n_embd, out_size=n_embd, bias=bias, backend=backend)
        self.dropout   = Dropout(p_dropout=p_dropout)

    def create_causal_mask(self, seq_len):
        """
        return a 1x1xTxt triangular causal mask for Q @ K^T (which will get broadcasted to BxHxTxT)
        """
        mask = -np.finfo(datatype).max * np.triu(np.ones((1, 1, seq_len, seq_len), dtype=datatype), 1)
        return tensor_from_numpy(mask, backend=self.backend)

    def q_kT_v(self, x):
        """Project hidden states to q, kT, v for self attention
        
        Args:
            x: embeddings or hidden states (B x S x D) from the previous decoder block

        Returns:
            q: The query vector used by multi-head attention (B x H x S x HD)
            kT: The transpose of the key vector used by multi-head attention (B x H x HD x S)
            v: The value vector used by multi-head attention (B x H x S x HD)
        """
        batch_size, seq_len, n_embd = x.shape
        x = x.view(batch_size * seq_len, n_embd)
        # Projects (B*S, D) -> (B*S, D)
        q = self.q_projection(x) 
        k = self.k_projection(x)
        v = self.v_projection(x)
        # Form MultiHead: (B, T, D = nH*hD) -> (B, T, nH, hD) -> (B, H, T, D) / (B, H, D, T)
        q = q.view(batch_size, seq_len, self.n_head, self.attn_hidden_dim).permute(0,2,1,3)
        kT = k.view(batch_size, seq_len, self.n_head, self.attn_hidden_dim).permute(0,2,3,1)
        v = v.view(batch_size, seq_len, self.n_head, self.attn_hidden_dim).permute(0,2,1,3)
        
        return q, kT, v
    
    def self_attention(self, q, kT, v):
        batch_size, num_head, queries_len, q_dim = q.shape
        _, _, k_dim, _ = kT.shape
        _, _, _, v_dim = v.shape
        assert q_dim == k_dim == v_dim
        result = None
        probs  = None
        
        ### BEGIN YOUR SOLUTION
        # Implicitly using Batch Mat Mul, which is seems to be implemented in minitorch
        q_k_t  = (q @ kT) / np.sqrt(q_dim) 
        if self.causal:
            q_k_t += self.create_causal_mask(queries_len)
        attn   = softmax(q_k_t, dim=3)
        probs  = self.dropout(attn)
        result = probs @ v # (B, H, T, d) bmm
        result = result.contiguous().permute(0, 2, 1, 3).contiguous().view(batch_size, queries_len, num_head * q_dim)
        ### END YOUR SOLUTION

        return result, probs

    def forward(self, x):
        batch_size, seq_len, n_embd = x.shape
        q, kT, v = self.q_kT_v(x)
        attn, _ = self.self_attention(q, kT, v)
        attn = attn.contiguous().view(batch_size * seq_len, n_embd)
        result = self.out_projection(attn)        
        result = result.contiguous().view(batch_size, seq_len, n_embd)
        return result


class FeedForward(Module):
    def __init__(self, n_embd: int, middle_dim: int, p_dropout: float=0.1, bias: bool=True, backend: TensorBackend=None):
        """Initialize the modules used by feedforward."""
        super().__init__()
        self.linear_in  = Linear(n_embd, middle_dim, bias=bias, backend=backend)
        self.linear_out = Linear(middle_dim, n_embd, bias=bias, backend=backend)
        self.dropout    = Dropout(p_dropout)

    def forward(self, x):
        """A full forward pass of the feedforward module.
        NOTE: We will need to turn x into (B * S, D) before linear and reshape back

        Args:
            x: outputs (B x S x D) of the first Add & Norm operation

        Returns:
            z: outputs (B x S x D) of the feedforward module

        Different from what you saw in class which uses ReLU as the activation,
        we are going to follow GPT-2 which uses GeLU. You should also apply
        self.dropout to the output.
        """
        ### BEGIN SOLUTION
        batch_size, seq_len, n_embd = x.shape
        x = GELU(self.linear_in(x.view(batch_size * seq_len, n_embd)))
        # Relu will be faster
        x = self.dropout(self.linear_out(x)).view(batch_size, seq_len, n_embd)
        ### END SOLUTION

        return x

class TransformerLayer(Module):
    def __init__(self, n_embd: int, n_head: int, p_dropout: float=0.1, ln_eps: float=1e-5, bias: bool=True, backend: TensorBackend=None):
        super().__init__()
        self.ln_1 = LayerNorm1d(dim=n_embd, eps=ln_eps, backend=backend)
        self.ln_2 = LayerNorm1d(dim=n_embd, eps=ln_eps, backend=backend)
        self.attention = MultiHeadAttention(n_embd=n_embd, n_head=n_head, p_dropout=p_dropout, backend=backend)
        self.ff   = FeedForward(n_embd=n_embd, middle_dim=256, p_dropout=p_dropout, bias=bias, backend=backend)

    def forward(self, x):
        """
        The forward function of a Transformer Layer for a PRENORM Transformer.
        Input: the hidden states from previous layers `x` with shape (batch_size, seq_len, x_dim)
        Ouput: the hidden states after the Transformer Layer `x` with shape (batch_size, seq_len, x_dim)
        """
        batch_size, seq_len, x_dim = x.shape
        
        ### BEGIN SOLUTION
        x = x + self.attention(self.ln_1(x.view(batch_size * seq_len, x_dim)).view(batch_size, seq_len, x_dim)) 
        x = x + self.ff(self.ln_2(x.view(batch_size * seq_len, x_dim)).view(batch_size, seq_len, x_dim))
        ### END SOLUTION

        return x


class DecoderLM(Module):
    def __init__(
        self, 
        n_vocab: int,
        n_embd: int,
        n_head: int,
        n_positions: int,
        n_layer: int,
        p_dropout: float=0.1,
        ln_eps: float=1e-5, 
        bias: bool=True,
        backend: TensorBackend=None
    ):
        super().__init__()
        self.backend             = backend
        self.n_embd              = n_embd
        self.n_vocab             = n_vocab

        self.token_embeddings    = Embedding(num_embeddings=n_vocab, embedding_dim=n_embd, backend=backend)
        self.position_embeddings = Embedding(num_embeddings=n_positions, embedding_dim=n_embd, backend=backend)
        self.t_layer_1  = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend)
        self.t_layer_2  = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend)
        self.t_layer_3  = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend)
        self.t_layer_4  = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend)
        self.dropout             = Dropout(p_dropout)
        self.ln                  = LayerNorm1d(dim=n_embd, eps=ln_eps, backend=backend)
        self.lm_head             = Linear(in_size=n_embd, out_size=n_vocab, bias=False, backend=backend)
    
    def forward(self, idx):
        """
        Args: 
            idx: input of shape (batch_size, seq_len)
        
        Returns: 
            logits: logits of shape (batch_size, seq_len, vocab_size)
        """
        
        batch_size, seq_len = idx.shape
        # Cannot initialize with numpy, need to implicitly broadcast
        pos = tensor([i for i in range(seq_len)], backend=self.backend).view(1, seq_len)

        ### BEGIN SOLUTION
        # (batch_size, seq_len, n_embd)
        tok_emb = self.token_embeddings(idx)
        # (1, seq_len, n_embd)
        pos_emb = self.position_embeddings(pos)
        # # Input
        x = tok_emb + pos_emb # Broadcasting
        ### MINITORCH LAYERS
        x = self.t_layer_1(x)
        x = self.t_layer_2(x)
        x = self.t_layer_3(x)
        x = self.t_layer_4(x)
        ###
        x = self.ln(x.view(batch_size * seq_len, self.n_embd))
        x = self.lm_head(x).view(batch_size, seq_len, self.n_vocab)
        ### END SOLUTION

        return x


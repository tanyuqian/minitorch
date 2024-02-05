# Assignment 2

## Conditional Generation tasks

Usually decoder-only architecture is not that suitable compared with seq2seq.

To make the setting as clean as possible (prevent padding tricks & special attention masks), the generation in this implementation is not batchified.

### Requirement
```
pip install nltk rouge-score fire datasets transformers sacrebleu
```

#### Install minitorch locally
```
pip install -e .
```

#### Compile cuda files
```
bash compile_cuda.sh
```
## Problem 0: cuBLAS Matrix Multiplication


## Problem 1: Adding Pow and Tanh
We're still missing a few important arithmetic operations for Transformers, namely element-wise (e-wise) power and element-wise tanh. 

### 1. Implementent the forward and backward functions for the Tanh and PowerScalar tensor function in `minitorch/tensor_functions.py`

Recall from lecture the structure of minitorch. Calling `.tanh()` on a tensor for example will call a Tensor Function defined in `tensor_functions.py`. These functions are implemented on the CudaKernelBackend, which execute the actual operations on the tensors.  

You should utilize `tanh_map` and `pow_scalar_zip`, which have already been added to the `TensorBackend` class, which your CudaKernelOps should then implement.

Don't forget to save the necessary values in the context in the forward pass for your backward pass when calculating the derivatives. 

Since we're taking e-wise tanh and power, your gradient calculation should be very simple.

### 2. Implement the power and tanh function in combine.cu.

Add the following snippet to your `__device__ float fn` function in `minitorch/combine.cu`
```
case POW: {
    // BEGIN YOUR SOLUTION
    return;
    // END YOUR SOLUTION
}
case TANH: {
    // BEGIN YOUR SOLUTION
    return;
    // END YOUR SOLUTION
}
```

Complete the Cuda code to support element-wise power and tanh. 


You can look up the relevant mathematical functions here: <a href="https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE" target="_blank">CUDA Math API</a>


### 3. Recompile your code with the bash command above.

### 4. Run the tests below.

The accompanying tests are in `tests/test_tensor_general.py`

Run the following to test an individual function eg.
```
python -m pytest -l -v -k "test_pow_1"
```
Run the following to test all parts to problem 0.
```
python -m pytest -l -v -m a2_1
```


## Problem 2: Implementing Tensor Functions

You will be implementing all the necessary functions and modules to implement a decoder-only transformer model. **PLEASE READ THE _IMPLEMENTATION DETAILS_ SECTION BEFORE STARTING** regarding advice for working with miniTorch.

Implement the GELU activation, logsumexp, one_hot, and softmax_loss functions in `minitorch/nn.py`
The accompanying tests are in `tests/test_nn.py`

Hints:
-  **one_hot**: Since MiniTorch doesn't support slicing/indexing with tensors, you'll want to utilize Numpy's eye function. You can use the .to_numpy() function for MiniTorch Tensors here. (Try to avoid using this in other functions because it's expensive.)

- **softmax_los**: You'll want to make use of your previously implemented one_hot function.

Run the following to test an individual function eg.
```
python -m pytest -l -v -k "test_gelu"
```
Run the following to test all the parts to Problem 2
```
python -m pytest -l -v -m a2_2
```


## Problem 3: Implementing Basic Modules
Implement the Embedding, Dropout, Linear, and LayerNorm1d modules in `minitorch/modules_basic.py`
The accompanying tests are in `tests/test_modules_basic.py`

Run the following to test an individual function eg.
```
python -m pytest -l -v -k "test_embedding"
```
Run the following to test question 1.1
```
python -m pytest -l -v -m a2_2
```


## Problem 4: Implementing a Decoder-only Transformer

Implement the MultiHeadAttention, FeedForward, TransformerLayer, and DecoderLM module in `minitorch/modules_transfomer.py`.
The accompanying tests are in `tests/test_modules_transformer.py`

Run the following to test an individual function eg.
```
python -m pytest -l -v -k "test_multihead_attention"
```
Run the following to test question 1.1
```
python -m pytest -l -v -m a2_3
```


## Problem 5

Implement a machine translation pipeline in `project/run_machine_translation.py`


## Implementation Details
 - Initializing parameters

When initializing weights in a Module, **always** wrap them with `Parameter(.)`, otherwise miniTorch will not update it.

 - Using `_from_numpy` functions

We've provided a new set of tensor initialization functions eg. `tensor_from_numpy`.
Feel free to use them in functions like one_hot, since minitorch doesn't support slicing, or other times **when you need numpy functions and minitorch doesn't support them**. Only initialize tensors with a new numpy nd.array (slicing is ok) ie. don't call new functions on them like .T before passing the nd.array into these functions. If you're getting errors, create a copy or deep copy of your numpy array, but you shouldn't need this.

 - Requiring Gradients

When you initialize parameters eg. in LayerNorm, **make sure you set the require_grad_ field** for parameters or tensors for which you'll need to update. 

 - Broadcasting - implicit broadcasting

Unlike numpy or torch, we don't have the `broadcast_to` function available. However, we do have _implicit broadcasting_. eg. given a tensors of shape (2, 2) and (1, 2), you can add the two tensors and the second tensor will be broadcasted to the first tensor using standard broadcasting rules. You will encounter this when building your modules, so keep this in mind if you ever feel like you need `broadcast_to`.

 - Contiguous Arrays

Some operations like view require arrays to be contiguous. Sometimes adding a .contiguous() may fix your error.

 - No sequential

 There is easy way to add sequential modules. **Do not put transformer layers in a list/iterable** and iterate through it in your forward function, because miniTorch will not recognize it

 - Always add backend

Always ensure your parameters are initialized with the correct backend (with your CudaKernelOps) to ensure they're run correctly.

 - Batch Matrix Multiplication

 We support batched matrix multiplication: 
 Given tensors A and B of shape (a, b, m, n) and (a, b, n, p), A @ B will be of shape (a, b, m, p),
 whereby matrices are multiplied elementwise across dimensions 0 and 1.

 - MiniTorch behavior when preserving dimensions

 - Possible errors


### Machine Translation (IWSLT-14)

```
python project/run_torch_machine_translation.py
```

|                            | BLEU | Running time (RTX-3090)     | Speed (RTX-3090)  |
|----------------------------|------|-----------------------------|-------------------|
| Transformer (Seq2seq)      | 34   |                             | 
| Ours (GPT2 with PyTorch)   | 27   | 34 mins / epoch * 10 epochs | 59K tokens / sec. |
| Ours (GPT2 with MiniTorch) |      |                             |

### Summarization (Gigaword, Deprecated)

```
python project/run_torch_conditional_generation.py --dataset_name gigaword --samples_per_epoch 200000
```
* `--samples_per_epoch 200000`: gigaword has a very large training set (3M), so here every epoch we only sample a part of it to save training time.  


|                            | Rouge-1 | Rouge-2 | Rouge-L | Running time (RTX-3090)     | Speed (RTX-3090)  |
|----------------------------|---------|---------|---------|---------------------------------------|-------------------|
| Transformer (Seq2seq)      |  37.57  |  18.90  |  34.69  |                                       |
| Ours (GPT2 with PyTorch)   | 33.84   | 15.06   | 31.31   | 19 mins / epoch * 10 epochs | 50K tokens / sec. |
| Ours (GPT2 with MiniTorch) |         |         |         |                                       |
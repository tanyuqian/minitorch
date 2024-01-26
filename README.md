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

### Implementing a decoder-only transfomer model in miniTorch

The modules and functions needed to be implemented can be found in 
* `modules_transformer.py` contains all transformer related modules including *DecoderLM*
* `modules_basic.py` contains Embedding, Sequential, Dropout, Linear, LayerNorm1d
* `nn.py` contains GELU, cross_entropy, 

The unit tests for each module can be found in `test_modules_transformer.py`.
Individual tests can be run with 
```
python -m pytest -l -v -k "test_transformer_layer"
```

All tests can be run with. 
Current works with `minitorch.TensorBackend(CudaKernelOps)` but may fail with FastOps.
```
python -m pytest -l -v -k test_modules_transformer.py
```

#### *NOTE*
The forward pass of the DecoderLM will work without any `__pow__` (**) operations (not supported yet)

#### *TODO*
* Add support for `__pow__`.
* (?) Add support for tanh to be used in GELU
* Verify A1 implementation of `var` in `tensor.py` in dim not None case. 
* Verify why tests fail with `TensorBackend(FastOps)` after rebasing onto Assignment 1
    Now running into `ZeroDivisionError: division by zero` but not before the rebase. 
    Commit `4f20c9fd3634812451132d4b6ac4c751cc7eee1a` has working FastOps.
* Verify correctness of reference modules.
* Train DecoderLM. 


#### *Possible Improvements*
* Add support for initialization with numpy array/torch tensor 

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
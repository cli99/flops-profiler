# Flops Profiler

[![PyPI](https://img.shields.io/pypi/v/flops-profiler.svg)](https://pypi.org/project/flops-profiler/)
[![Read the Docs](https://readthedocs.org/projects/flops-profiler/badge/)](https://flops-profiler.readthedocs.io/)
[![Tests](https://github.com/cli99/flops-profiler/workflows/tests/badge.svg)](https://github.com/cli99/flops-profiler/actions?workflow=tests)
[![Codecov](https://codecov.io/gh/cli99/flops-profiler/branch/main/graph/badge.svg)](https://codecov.io/gh/cli99/flops-profiler)
[![GitHub license](https://img.shields.io/github/license/cli99/flops-profiler)](https://github.com/cli99/flops-profiler/blob/main/LICENSE)

> Measures the parameters, latency, and floating-point operations of PyTorch model.

- [Flops Profiler](#flops-profiler)
  - [Install](#install)
  - [Overview](#overview)
  - [Examples](#examples)
  - [Flops Measurement](#flops-measurement)
  - [Multi-device, Multi-node, Data Parallelism, and Model Parallelism](#multi-device-multi-node-data-parallelism-and-model-parallelism)
  - [Usage](#usage)
    - [In Model Inference](#in-model-inference)
      - [Example: AlexNet](#example-alexnet)
    - [In Model Training Workflow](#in-model-training-workflow)
      - [Example Training Workflow](#example-training-workflow)

Similar to  [DeepSpeed Flops Profiler](https://github.com/microsoft/DeepSpeed) but more verbose, runs on both CPU and GPU, and explicitly shows all intra-module functional information at module level.

Note that the latency measurement on GPU requires `torch.cuda.synchronize()` and incurs timing overhead: the timings of the funtionals (topper modules' functional information are direct aggregations of the lower module's) are accurate while the module timings (captured with forward hooks) have overhead. Check the code for timing details.

## Install

Install the flops profiler by

```sh
pip install flops-profiler
```

## Overview

Effective use of hardware resources is critical to good performance, but performance inefficiency in existing implementations for large-scale model training and inference are often hard to spot and attributed to specific module components. The Flops Profiler helps users easily measure both the model training/inference speed (latency, throughput) and efficiency (floating-point operations per second, i.e., FLOPS) of a model and its submodules, with an eye towards eliminating inefficiencies in existing implementations.

Below is an example output for BERT-base on an A6000 GPU with batch size `1` and sequence length `128` (see [bert.py](examples/bert.py)):

```shell
MLFlow does not exist. Disabling MLFlow logging

-------------------------- Flops Profiler --------------------------
Profile Summary at step 3:
Notations:
data parallel size (dp_size), model parallel size(mp_size),
number of parameters (params), number of multiply-accumulate operations(MACs),
number of floating-point operations (flops), floating-point operations per second (FLOPS),
fwd latency (forward propagation latency), bwd latency (backward propagation latency),
step (weights update latency), iter latency (sum of fwd, bwd and step latency)

params per device:                                            109.48 M
params of model = params per device * mp_size:                109.48 M
fwd MACs per device:                                          11.17 GMACs
fwd flops per device:                                         22.36 G
fwd flops of model = fwd flops per device * mp_size:          22.36 G
fwd latency:                                                  12.22 ms
fwd FLOPS per device = fwd flops per device / fwd latency:    1.83 TFLOPS

----------------------------- Aggregated Profile per Device -----------------------------
Top 10 modules in terms of params, flops, MACs or duration at different model depths:
depth 0:
    params      - {'BertModel': '109.48 M'}
    flops       - {'BertModel': '22.36 G'}
    MACs        - {'BertModel': '11.17 GMACs'}
    fwd latency - {'BertModel': '12.22 ms'}
depth 1:
    params      - {'BertEncoder': '85.05 M', 'BertEmbeddings': '23.84 M', 'BertPooler': '590.59 k'}
    flops       - {'BertEncoder': '22.36 G', 'BertPooler': '1.18 M', 'BertEmbeddings': '491.52 K'}
    MACs        - {'BertEncoder': '11.17 GMACs', 'BertPooler': '589.82 KMACs', 'BertEmbeddings': '0 MACs'}
    fwd latency - {'BertEncoder': '11.93 ms', 'BertEmbeddings': '194.79 us', 'BertPooler': '92.98 us'}
depth 2:
    params      - {'ModuleList': '85.05 M', 'Embedding': '23.84 M', 'Linear': '590.59 k', 'LayerNorm': '1.54 k', 'Dropout': '0', 'Tanh': '0'}
    flops       - {'ModuleList': '22.36 G', 'Linear': '1.18 M', 'LayerNorm': '491.52 K', 'Embedding': '0', 'Dropout': '0', 'Tanh': '0'}
    MACs        - {'ModuleList': '11.17 GMACs', 'Linear': '589.82 KMACs', 'Embedding': '0 MACs', 'LayerNorm': '0 MACs', 'Dropout': '0 MACs', 'Tanh': '0 MACs'}
    fwd latency - {'ModuleList': '11.85 ms', 'Embedding': '86.07 us', 'Linear': '43.87 us', 'LayerNorm': '31.71 us', 'Tanh': '20.74 us', 'Dropout': '11.68 us'}
depth 3:
    params      - {'BertLayer': '85.05 M'}
    flops       - {'BertLayer': '22.36 G'}
    MACs        - {'BertLayer': '11.17 GMACs'}
    fwd latency - {'BertLayer': '11.85 ms'}
depth 4:
    params      - {'BertAttention': '28.37 M', 'BertIntermediate': '28.35 M', 'BertOutput': '28.34 M'}
    flops       - {'BertAttention': '7.86 G', 'BertOutput': '7.25 G', 'BertIntermediate': '7.25 G'}
    MACs        - {'BertAttention': '3.93 GMACs', 'BertIntermediate': '3.62 GMACs', 'BertOutput': '3.62 GMACs'}
    fwd latency - {'BertAttention': '8.59 ms', 'BertOutput': '1.47 ms', 'BertIntermediate': '1.23 ms'}
depth 5:
    params      - {'Linear': '56.67 M', 'BertSelfAttention': '21.26 M', 'BertSelfOutput': '7.11 M', 'LayerNorm': '18.43 k', 'GELUActivation': '0', 'Dropout': '0'}
    flops       - {'Linear': '14.5 G', 'BertSelfAttention': '6.04 G', 'BertSelfOutput': '1.82 G', 'LayerNorm': '5.9 M', 'GELUActivation': '0', 'Dropout': '0'}
    MACs        - {'Linear': '7.25 GMACs', 'BertSelfAttention': '3.02 GMACs', 'BertSelfOutput': '905.97 MMACs', 'GELUActivation': '0 MACs', 'LayerNorm': '0 MACs', 'Dropout': '0 MACs'}
    fwd latency - {'BertSelfAttention': '5.6 ms', 'BertSelfOutput': '2.79 ms', 'Linear': '1.25 ms', 'LayerNorm': '404.6 us', 'GELUActivation': '338.32 us', 'Dropout': '123.5 us'}

------------------------------ Detailed Profile per Device ------------------------------
Each module profile is listed after its name in the following order:
params, percentage of total params, MACs, percentage of total MACs, fwd latency, percentage of total fwd latency, fwd FLOPS

Note: 1. A module can have torch.nn.module or torch.nn.functional to compute logits (e.g. CrossEntropyLoss). They are not counted as submodules, thus not to be printed out. However they make up the difference between a parent's MACs (or latency) and the sum of its submodules'.
2. Number of floating-point operations is a theoretical estimation, thus FLOPS computed using that could be larger than the maximum system throughput.
3. The fwd latency listed in the top module's profile is directly captured at the module forward function in PyTorch.

BertModel(
  module = {'param': '109.48 M', 'flops': '22.36 G', 'macs': '11.17 GMACs', 'duration': '12.22 ms', 'FLOPS': '1.83 TFLOPS', 'params%': '100.00%', 'flops%': '100.00%', 'macs%': '100.00%', 'duration%': '100.00%'}, functionals = {'embedding': {'flops': '0', 'macs': '0 MACs', 'duration': '45.3 us', 'FLOPS': '0.0 FLOPS', 'flops%': '0.00%', 'macs%': '0.00%', 'duration%/allfuncs': '0.71%', 'duration%/e2e': '0.37%'}, 'layer_norm': {'flops': '12.29 M', 'macs': '0 MACs', 'duration': '502.11 us', 'FLOPS': '24.47 GFLOPS', 'flops%': '0.05%', 'macs%': '0.00%', 'duration%/allfuncs': '7.86%', 'duration%/e2e': '4.11%'}, 'matmul': {'flops': '603.98 M', 'macs': '301.99 MMACs', 'duration': '759.84 us', 'FLOPS': '794.88 GFLOPS', 'flops%': '2.70%', 'macs%': '2.70%', 'duration%/allfuncs': '11.89%', 'duration%/e2e': '6.22%'}, 'softmax': {'flops': '2.36 M', 'macs': '0 MACs', 'duration': '129.94 us', 'FLOPS': '18.16 GFLOPS', 'flops%': '0.01%', 'macs%': '0.00%', 'duration%/allfuncs': '2.03%', 'duration%/e2e': '1.06%'}, 'linear': {'flops': '21.74 G', 'macs': '10.87 GMACs', 'duration': '4.95 ms', 'FLOPS': '4.39 TFLOPS', 'flops%': '97.23%', 'macs%': '97.30%', 'duration%/allfuncs': '77.51%', 'duration%/e2e': '40.53%'}}, functionals_duration = 6.39 ms,
  (embeddings): BertEmbeddings(
    module = {'param': '23.84 M', 'flops': '491.52 K', 'macs': '0 MACs', 'duration': '194.79 us', 'FLOPS': '2.52 GFLOPS', 'params%': '21.77%', 'flops%': '0.00%', 'macs%': '0.00%', 'duration%': '1.59%'}, functionals = {'embedding': {'flops': '0', 'macs': '0 MACs', 'duration': '45.3 us', 'FLOPS': '0.0 FLOPS', 'flops%': '0.00%', 'macs%': '0.00%', 'duration%/allfuncs': '0.71%', 'duration%/e2e': '0.37%'}, 'layer_norm': {'flops': '491.52 K', 'macs': '0 MACs', 'duration': '18.36 us', 'FLOPS': '26.77 GFLOPS', 'flops%': '0.00%', 'macs%': '0.00%', 'duration%/allfuncs': '0.29%', 'duration%/e2e': '0.15%'}}, functionals_duration = 63.66 us,
    (word_embeddings): Embedding(module = {'param': '23.44 M', 'flops': '0', 'macs': '0 MACs', 'duration': '37.91 us', 'FLOPS': '0.0 FLOPS', 'params%': '21.41%', 'flops%': '0.00%', 'macs%': '0.00%', 'duration%': '0.31%'}, functionals = {'embedding': {'flops': '0', 'macs': '0 MACs', 'duration': '19.55 us', 'FLOPS': '0.0 FLOPS', 'flops%': '0.00%', 'macs%': '0.00%', 'duration%/allfuncs': '0.31%', 'duration%/e2e': '0.16%'}}, functionals_duration = 19.55 us, 30522, 768, padding_idx=0)
    (position_embeddings): Embedding(module = {'param': '393.22 k', 'flops': '0', 'macs': '0 MACs', 'duration': '24.08 us', 'FLOPS': '0.0 FLOPS', 'params%': '0.36%', 'flops%': '0.00%', 'macs%': '0.00%', 'duration%': '0.20%'}, functionals = {'embedding': {'flops': '0', 'macs': '0 MACs', 'duration': '13.11 us', 'FLOPS': '0.0 FLOPS', 'flops%': '0.00%', 'macs%': '0.00%', 'duration%/allfuncs': '0.21%', 'duration%/e2e': '0.11%'}}, functionals_duration = 13.11 us, 512, 768)
    (token_type_embeddings): Embedding(module = {'param': '1.54 k', 'flops': '0', 'macs': '0 MACs', 'duration': '24.08 us', 'FLOPS': '0.0 FLOPS', 'params%': '0.00%', 'flops%': '0.00%', 'macs%': '0.00%', 'duration%': '0.20%'}, functionals = {'embedding': {'flops': '0', 'macs': '0 MACs', 'duration': '12.64 us', 'FLOPS': '0.0 FLOPS', 'flops%': '0.00%', 'macs%': '0.00%', 'duration%/allfuncs': '0.20%', 'duration%/e2e': '0.10%'}}, functionals_duration = 12.64 us, 2, 768)
    (LayerNorm): LayerNorm(module = {'param': '1.54 k', 'flops': '491.52 K', 'macs': '0 MACs', 'duration': '31.71 us', 'FLOPS': '15.5 GFLOPS', 'params%': '0.00%', 'flops%': '0.00%', 'macs%': '0.00%', 'duration%': '0.26%'}, functionals = {'layer_norm': {'flops': '491.52 K', 'macs': '0 MACs', 'duration': '18.36 us', 'FLOPS': '26.77 GFLOPS', 'flops%': '0.00%', 'macs%': '0.00%', 'duration%/allfuncs': '0.29%', 'duration%/e2e': '0.15%'}}, functionals_duration = 18.36 us, (768,), eps=1e-12, elementwise_affine=True)
    (dropout): Dropout(module = {'param': '0', 'flops': '0', 'macs': '0 MACs', 'duration': '11.68 us', 'FLOPS': '0.0 FLOPS', 'params%': '0.00%', 'flops%': '0.00%', 'macs%': '0.00%', 'duration%': '0.10%'}, functionals = {}, functionals_duration = 0.0, p=0.1, inplace=False)
  )
  (encoder): BertEncoder(
    module = {'param': '85.05 M', 'flops': '22.36 G', 'macs': '11.17 GMACs', 'duration': '11.93 ms', 'FLOPS': '1.87 TFLOPS', 'params%': '77.69%', 'flops%': '99.99%', 'macs%': '99.99%', 'duration%': '97.64%'}, functionals = {'matmul': {'flops': '603.98 M', 'macs': '301.99 MMACs', 'duration': '759.84 us', 'FLOPS': '794.88 GFLOPS', 'flops%': '2.70%', 'macs%': '2.70%', 'duration%/allfuncs': '11.89%', 'duration%/e2e': '6.22%'}, 'softmax': {'flops': '2.36 M', 'macs': '0 MACs', 'duration': '129.94 us', 'FLOPS': '18.16 GFLOPS', 'flops%': '0.01%', 'macs%': '0.00%', 'duration%/allfuncs': '2.03%', 'duration%/e2e': '1.06%'}, 'linear': {'flops': '21.74 G', 'macs': '10.87 GMACs', 'duration': '4.92 ms', 'FLOPS': '4.42 TFLOPS', 'flops%': '97.23%', 'macs%': '97.29%', 'duration%/allfuncs': '77.05%', 'duration%/e2e': '40.29%'}, 'layer_norm': {'flops': '11.8 M', 'macs': '0 MACs', 'duration': '483.75 us', 'FLOPS': '24.39 GFLOPS', 'flops%': '0.05%', 'macs%': '0.00%', 'duration%/allfuncs': '7.57%', 'duration%/e2e': '3.96%'}}, functionals_duration = 6.3 ms,
    (layer): ModuleList(
      module = {'param': '85.05 M', 'flops': '22.36 G', 'macs': '11.17 GMACs', 'duration': '11.85 ms', 'FLOPS': '1.89 TFLOPS', 'params%': '77.69%', 'flops%': '99.99%', 'macs%': '99.99%', 'duration%': '97.00%'}, functionals = {'matmul': {'flops': '603.98 M', 'macs': '301.99 MMACs', 'duration': '759.84 us', 'FLOPS': '794.88 GFLOPS', 'flops%': '2.70%', 'macs%': '2.70%', 'duration%/allfuncs': '11.89%', 'duration%/e2e': '6.22%'}, 'softmax': {'flops': '2.36 M', 'macs': '0 MACs', 'duration': '129.94 us', 'FLOPS': '18.16 GFLOPS', 'flops%': '0.01%', 'macs%': '0.00%', 'duration%/allfuncs': '2.03%', 'duration%/e2e': '1.06%'}, 'linear': {'flops': '21.74 G', 'macs': '10.87 GMACs', 'duration': '4.92 ms', 'FLOPS': '4.42 TFLOPS', 'flops%': '97.23%', 'macs%': '97.29%', 'duration%/allfuncs': '77.05%', 'duration%/e2e': '40.29%'}, 'layer_norm': {'flops': '11.8 M', 'macs': '0 MACs', 'duration': '483.75 us', 'FLOPS': '24.39 GFLOPS', 'flops%': '0.05%', 'macs%': '0.00%', 'duration%/allfuncs': '7.57%', 'duration%/e2e': '3.96%'}}, functionals_duration = 6.3 ms,
      (0): BertLayer(
        module = {'param': '7.09 M', 'flops': '1.86 G', 'macs': '931.14 MMACs', 'duration': '997.3 us', 'FLOPS': '1.87 TFLOPS', 'params%': '6.47%', 'flops%': '8.33%', 'macs%': '8.33%', 'duration%': '8.16%'}, functionals = {'matmul': {'flops': '50.33 M', 'macs': '25.17 MMACs', 'duration': '63.42 us', 'FLOPS': '793.63 GFLOPS', 'flops%': '0.23%', 'macs%': '0.23%', 'duration%/allfuncs': '0.99%', 'duration%/e2e': '0.52%'}, 'softmax': {'flops': '196.61 K', 'macs': '0 MACs', 'duration': '10.97 us', 'FLOPS': '17.93 GFLOPS', 'flops%': '0.00%', 'macs%': '0.00%', 'duration%/allfuncs': '0.17%', 'duration%/e2e': '0.09%'}, 'linear': {'flops': '1.81 G', 'macs': '905.97 MMACs', 'duration': '415.8 us', 'FLOPS': '4.36 TFLOPS', 'flops%': '8.10%', 'macs%': '8.11%', 'duration%/allfuncs': '6.51%', 'duration%/e2e': '3.40%'}, 'layer_norm': {'flops': '983.04 K', 'macs': '0 MACs', 'duration': '39.58 us', 'FLOPS': '24.84 GFLOPS', 'flops%': '0.00%', 'macs%': '0.00%', 'duration%/allfuncs': '0.62%', 'duration%/e2e': '0.32%'}}, functionals_duration = 529.77 us,
        (attention): BertAttention(
          module = {'param': '2.36 M', 'flops': '655.0 M', 'macs': '327.16 MMACs', 'duration': '742.44 us', 'FLOPS': '882.23 GFLOPS', 'params%': '2.16%', 'flops%': '2.93%', 'macs%': '2.93%', 'duration%': '6.08%'}, functionals = {'matmul': {'flops': '50.33 M', 'macs': '25.17 MMACs', 'duration': '63.42 us', 'FLOPS': '793.63 GFLOPS', 'flops%': '0.23%', 'macs%': '0.23%', 'duration%/allfuncs': '0.99%', 'duration%/e2e': '0.52%'}, 'softmax': {'flops': '196.61 K', 'macs': '0 MACs', 'duration': '10.97 us', 'FLOPS': '17.93 GFLOPS', 'flops%': '0.00%', 'macs%': '0.00%', 'duration%/allfuncs': '0.17%', 'duration%/e2e': '0.09%'}, 'linear': {'flops': '603.98 M', 'macs': '301.99 MMACs', 'duration': '348.57 us', 'FLOPS': '1.73 TFLOPS', 'flops%': '2.70%', 'macs%': '2.70%', 'duration%/allfuncs': '5.46%', 'duration%/e2e': '2.85%'}, 'layer_norm': {'flops': '491.52 K', 'macs': '0 MACs', 'duration': '21.93 us', 'FLOPS': '22.41 GFLOPS', 'flops%': '0.00%', 'macs%': '0.00%', 'duration%/allfuncs': '0.34%', 'duration%/e2e': '0.18%'}}, functionals_duration = 444.89 us,
          (self): BertSelfAttention(
            module = {'param': '1.77 M', 'flops': '503.51 M', 'macs': '251.66 MMACs', 'duration': '487.09 us', 'FLOPS': '1.03 TFLOPS', 'params%': '1.62%', 'flops%': '2.25%', 'macs%': '2.25%', 'duration%': '3.99%'}, functionals = {'matmul': {'flops': '50.33 M', 'macs': '25.17 MMACs', 'duration': '63.42 us', 'FLOPS': '793.63 GFLOPS', 'flops%': '0.23%', 'macs%': '0.23%', 'duration%/allfuncs': '0.99%', 'duration%/e2e': '0.52%'}, 'softmax': {'flops': '196.61 K', 'macs': '0 MACs', 'duration': '10.97 us', 'FLOPS': '17.93 GFLOPS', 'flops%': '0.00%', 'macs%': '0.00%', 'duration%/allfuncs': '0.17%', 'duration%/e2e': '0.09%'}, 'linear': {'flops': '452.98 M', 'macs': '226.49 MMACs', 'duration': '209.81 us', 'FLOPS': '2.16 TFLOPS', 'flops%': '2.03%', 'macs%': '2.03%', 'duration%/allfuncs': '3.28%', 'duration%/e2e': '1.72%'}}, functionals_duration = 284.19 us,
            (query): Linear(module = {'param': '590.59 k', 'flops': '150.99 M', 'macs': '75.5 MMACs', 'duration': '168.32 us', 'FLOPS': '897.05 GFLOPS', 'params%': '0.54%', 'flops%': '0.68%', 'macs%': '0.68%', 'duration%': '1.38%'}, functionals = {'linear': {'flops': '150.99 M', 'macs': '75.5 MMACs', 'duration': '149.25 us', 'FLOPS': '1.01 TFLOPS', 'flops%': '0.68%', 'macs%': '0.68%', 'duration%/allfuncs': '2.34%', 'duration%/e2e': '1.22%'}}, functionals_duration = 149.25 us, in_features=768, out_features=768, bias=True)
            (key): Linear(module = {'param': '590.59 k', 'flops': '150.99 M', 'macs': '75.5 MMACs', 'duration': '45.3 us', 'FLOPS': '3.33 TFLOPS', 'params%': '0.54%', 'flops%': '0.68%', 'macs%': '0.68%', 'duration%': '0.37%'}, functionals = {'linear': {'flops': '150.99 M', 'macs': '75.5 MMACs', 'duration': '31.47 us', 'FLOPS': '4.8 TFLOPS', 'flops%': '0.68%', 'macs%': '0.68%', 'duration%/allfuncs': '0.49%', 'duration%/e2e': '0.26%'}}, functionals_duration = 31.47 us, in_features=768, out_features=768, bias=True)
            (value): Linear(module = {'param': '590.59 k', 'flops': '150.99 M', 'macs': '75.5 MMACs', 'duration': '41.72 us', 'FLOPS': '3.62 TFLOPS', 'params%': '0.54%', 'flops%': '0.68%', 'macs%': '0.68%', 'duration%': '0.34%'}, functionals = {'linear': {'flops': '150.99 M', 'macs': '75.5 MMACs', 'duration': '29.09 us', 'FLOPS': '5.19 TFLOPS', 'flops%': '0.68%', 'macs%': '0.68%', 'duration%/allfuncs': '0.46%', 'duration%/e2e': '0.24%'}}, functionals_duration = 29.09 us, in_features=768, out_features=768, bias=True)
            (dropout): Dropout(module = {'param': '0', 'flops': '0', 'macs': '0 MACs', 'duration': '13.11 us', 'FLOPS': '0.0 FLOPS', 'params%': '0.00%', 'flops%': '0.00%', 'macs%': '0.00%', 'duration%': '0.11%'}, functionals = {}, functionals_duration = 0.0, p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            module = {'param': '592.13 k', 'flops': '151.49 M', 'macs': '75.5 MMACs', 'duration': '237.7 us', 'FLOPS': '637.29 GFLOPS', 'params%': '0.54%', 'flops%': '0.68%', 'macs%': '0.68%', 'duration%': '1.95%'}, functionals = {'linear': {'flops': '150.99 M', 'macs': '75.5 MMACs', 'duration': '138.76 us', 'FLOPS': '1.09 TFLOPS', 'flops%': '0.68%', 'macs%': '0.68%', 'duration%/allfuncs': '2.17%', 'duration%/e2e': '1.14%'}, 'layer_norm': {'flops': '491.52 K', 'macs': '0 MACs', 'duration': '21.93 us', 'FLOPS': '22.41 GFLOPS', 'flops%': '0.00%', 'macs%': '0.00%', 'duration%/allfuncs': '0.34%', 'duration%/e2e': '0.18%'}}, functionals_duration = 160.69 us,
            (dense): Linear(module = {'param': '590.59 k', 'flops': '150.99 M', 'macs': '75.5 MMACs', 'duration': '155.45 us', 'FLOPS': '971.35 GFLOPS', 'params%': '0.54%', 'flops%': '0.68%', 'macs%': '0.68%', 'duration%': '1.27%'}, functionals = {'linear': {'flops': '150.99 M', 'macs': '75.5 MMACs', 'duration': '138.76 us', 'FLOPS': '1.09 TFLOPS', 'flops%': '0.68%', 'macs%': '0.68%', 'duration%/allfuncs': '2.17%', 'duration%/e2e': '1.14%'}}, functionals_duration = 138.76 us, in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm(module = {'param': '1.54 k', 'flops': '491.52 K', 'macs': '0 MACs', 'duration': '35.29 us', 'FLOPS': '13.93 GFLOPS', 'params%': '0.00%', 'flops%': '0.00%', 'macs%': '0.00%', 'duration%': '0.29%'}, functionals = {'layer_norm': {'flops': '491.52 K', 'macs': '0 MACs', 'duration': '21.93 us', 'FLOPS': '22.41 GFLOPS', 'flops%': '0.00%', 'macs%': '0.00%', 'duration%/allfuncs': '0.34%', 'duration%/e2e': '0.18%'}}, functionals_duration = 21.93 us, (768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(module = {'param': '0', 'flops': '0', 'macs': '0 MACs', 'duration': '10.73 us', 'FLOPS': '0.0 FLOPS', 'params%': '0.00%', 'flops%': '0.00%', 'macs%': '0.00%', 'duration%': '0.09%'}, functionals = {}, functionals_duration = 0.0, p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          module = {'param': '2.36 M', 'flops': '603.98 M', 'macs': '301.99 MMACs', 'duration': '85.83 us', 'FLOPS': '7.04 TFLOPS', 'params%': '2.16%', 'flops%': '2.70%', 'macs%': '2.70%', 'duration%': '0.70%'}, functionals = {'linear': {'flops': '603.98 M', 'macs': '301.99 MMACs', 'duration': '35.76 us', 'FLOPS': '16.89 TFLOPS', 'flops%': '2.70%', 'macs%': '2.70%', 'duration%/allfuncs': '0.56%', 'duration%/e2e': '0.29%'}}, functionals_duration = 35.76 us,
          (dense): Linear(module = {'param': '2.36 M', 'flops': '603.98 M', 'macs': '301.99 MMACs', 'duration': '50.54 us', 'FLOPS': '11.95 TFLOPS', 'params%': '2.16%', 'flops%': '2.70%', 'macs%': '2.70%', 'duration%': '0.41%'}, functionals = {'linear': {'flops': '603.98 M', 'macs': '301.99 MMACs', 'duration': '35.76 us', 'FLOPS': '16.89 TFLOPS', 'flops%': '2.70%', 'macs%': '2.70%', 'duration%/allfuncs': '0.56%', 'duration%/e2e': '0.29%'}}, functionals_duration = 35.76 us, in_features=768, out_features=3072, bias=True)
          (intermediate_act_fn): GELUActivation(module = {'param': '0', 'flops': '0', 'macs': '0 MACs', 'duration': '19.55 us', 'FLOPS': '0.0 FLOPS', 'params%': '0.00%', 'flops%': '0.00%', 'macs%': '0.00%', 'duration%': '0.16%'}, functionals = {}, functionals_duration = 0.0, )
        )
        (output): BertOutput(
          module = {'param': '2.36 M', 'flops': '604.47 M', 'macs': '301.99 MMACs', 'duration': '118.73 us', 'FLOPS': '5.09 TFLOPS', 'params%': '2.16%', 'flops%': '2.70%', 'macs%': '2.70%', 'duration%': '0.97%'}, functionals = {'linear': {'flops': '603.98 M', 'macs': '301.99 MMACs', 'duration': '31.47 us', 'FLOPS': '19.19 TFLOPS', 'flops%': '2.70%', 'macs%': '2.70%', 'duration%/allfuncs': '0.49%', 'duration%/e2e': '0.26%'}, 'layer_norm': {'flops': '491.52 K', 'macs': '0 MACs', 'duration': '17.64 us', 'FLOPS': '27.86 GFLOPS', 'flops%': '0.00%', 'macs%': '0.00%', 'duration%/allfuncs': '0.28%', 'duration%/e2e': '0.14%'}}, functionals_duration = 49.11 us,
          (dense): Linear(module = {'param': '2.36 M', 'flops': '603.98 M', 'macs': '301.99 MMACs', 'duration': '44.58 us', 'FLOPS': '13.55 TFLOPS', 'params%': '2.16%', 'flops%': '2.70%', 'macs%': '2.70%', 'duration%': '0.36%'}, functionals = {'linear': {'flops': '603.98 M', 'macs': '301.99 MMACs', 'duration': '31.47 us', 'FLOPS': '19.19 TFLOPS', 'flops%': '2.70%', 'macs%': '2.70%', 'duration%/allfuncs': '0.49%', 'duration%/e2e': '0.26%'}}, functionals_duration = 31.47 us, in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm(module = {'param': '1.54 k', 'flops': '491.52 K', 'macs': '0 MACs', 'duration': '30.04 us', 'FLOPS': '16.36 GFLOPS', 'params%': '0.00%', 'flops%': '0.00%', 'macs%': '0.00%', 'duration%': '0.25%'}, functionals = {'layer_norm': {'flops': '491.52 K', 'macs': '0 MACs', 'duration': '17.64 us', 'FLOPS': '27.86 GFLOPS', 'flops%': '0.00%', 'macs%': '0.00%', 'duration%/allfuncs': '0.28%', 'duration%/e2e': '0.14%'}}, functionals_duration = 17.64 us, (768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(module = {'param': '0', 'flops': '0', 'macs': '0 MACs', 'duration': '10.49 us', 'FLOPS': '0.0 FLOPS', 'params%': '0.00%', 'flops%': '0.00%', 'macs%': '0.00%', 'duration%': '0.09%'}, functionals = {}, functionals_duration = 0.0, p=0.1, inplace=False)
        )
      )
      ...
      (11): BertLayer(...)
    )
  )
  (pooler): BertPooler(
    module = {'param': '590.59 k', 'flops': '1.18 M', 'macs': '589.82 KMACs', 'duration': '92.98 us', 'FLOPS': '12.69 GFLOPS', 'params%': '0.54%', 'flops%': '0.01%', 'macs%': '0.01%', 'duration%': '0.76%'}, functionals = {'linear': {'flops': '1.18 M', 'macs': '589.82 KMACs', 'duration': '29.09 us', 'FLOPS': '40.56 GFLOPS', 'flops%': '0.01%', 'macs%': '0.01%', 'duration%/allfuncs': '0.46%', 'duration%/e2e': '0.24%'}}, functionals_duration = 29.09 us,
    (dense): Linear(module = {'param': '590.59 k', 'flops': '1.18 M', 'macs': '589.82 KMACs', 'duration': '43.87 us', 'FLOPS': '26.89 GFLOPS', 'params%': '0.54%', 'flops%': '0.01%', 'macs%': '0.01%', 'duration%': '0.36%'}, functionals = {'linear': {'flops': '1.18 M', 'macs': '589.82 KMACs', 'duration': '29.09 us', 'FLOPS': '40.56 GFLOPS', 'flops%': '0.01%', 'macs%': '0.01%', 'duration%/allfuncs': '0.46%', 'duration%/e2e': '0.24%'}}, functionals_duration = 29.09 us, in_features=768, out_features=768, bias=True)
    (activation): Tanh(module = {'param': '0', 'flops': '0', 'macs': '0 MACs', 'duration': '20.74 us', 'FLOPS': '0.0 FLOPS', 'params%': '0.00%', 'flops%': '0.00%', 'macs%': '0.00%', 'duration%': '0.17%'}, functionals = {}, functionals_duration = 0.0, )
  )
)
------------------------------------------------------------------------------
Number of flops:                22.36 G
Number of MACs:                 11.17 GMACs
Number of parameters:           109.48 M
```

In the summary profile, the Flops Profiler outputs the number of parameters, floating-point operations (flops), FLOPS, latency, and throughput in samples/second of the model. This profile shows how much performance gap (compared to the peak hardware performance) the current model execution has and helps users tune the training or inference setup (e.g., hyperparameters, data parallelism, model parallelism, system configurations, etc.) for better performance.

The Flops Profiler also measures significant modules at different model depths (aggregated profile) and module-specific profile in the model architecture (detailed profile). With these profiles users one can understand how each layer or submodule contributes to the overall model complexity/performance. Then users can adjust or refactor the model design to achieve better performance.

## Examples
 * [bert.py](examples/bert.py)
 * [t5.py](examples/t5.py)
 * [vision.py](examples/vision.py)
 * [gpt2.py](examples/gpt2.py)

## Flops Measurement

Similar to existing flops calculation tools or methods, the Flops Profiler measures the flops of the forward pass of a module and the flops of the backward pass is estimated as `2` times of that of the forward pass.
Different from the PyTorch profiler which calculates the flops of PyTorch operators, the Flops Profiler measures the flops within modules in a model and provides more insights to the users about the model execution.
The flops estimation is partly inspired by [ptflops](https://github.com/sovrasov/flops-counter.pytorch) with the major difference being that the Flops Profiler not only supports flops computation directly at module level, but can also capture ```torch.nn.functional``` invoked in a module to estimate the flops.
Thus the Flops Profiler allows for customized modules in the model, e.g., ```ParallelTransformerLayerworks, ParallelSelfAttention, RowParallelLinear, etc.``` in [Megatron-LM](https://github.com/NVIDIA/Megatron-LM). This is in contrast to ptflops which requires users to write customized flops calculation functions for each customized module.

## Multi-device, Multi-node, Data Parallelism, and Model Parallelism

The Flops Profiler outputs the **PER DEVICE** profile. When initialized with a distributed runtime where world size(`world_size`), data parallel size(`dp_world_size`), and model parallel size(`mp_world_size`) are defined, the profiler uses this information; otherwise they are default to `1` in flops calcuation. See `ds_engine` in source code as a reference.                                       1
Note that for models running on multi-device or multi-node, only change of the model parallelism (e.g. ```--model-parallel-size``` in [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)) affects the number of flops and parameters profiled, i.e.,
`model_parallel_size * flops = total_flops` and `model_parallel_size * parameters = total_parameters`.
The data parallel size or world size (related to the number of GPUs or nodes) does not affect the per device profile.

## Usage
### In Model Inference

To profile a trained model in inference, we use the `get_model_profile` function. If the inference is involed in more than just a `forward` function of the model, for example, `model.generate()`, we can use the `start_profile`, `stop_profile`, and `end_profile` to capture the higher-level function (similar to the training use case); Or pass in `mode='generate'` when calling `get_model_profile`.

Examples are given below.

#### Example: AlexNet

The following example shows how to profile AlexNet using the Flops Profiler.

```python
import torchvision.models as models
import torch
from flops_profiler import get_model_profile

with torch.cuda.device(0):
    model = models.alexnet()
    batch_size = 256
    flops, macs, params = get_model_profile(model=model, # model
                                    input_shape=(batch_size, 3, 224, 224), # input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
                                    args=None, # list of positional arguments to the model.
                                    kwargs=None, # dictionary of keyword arguments to the model.
                                    print_profile=True, # prints the model graph with the measured profile attached to each module
                                    detailed=True, # print the detailed profile
                                    module_depth=-1, # depth into the nested modules, with -1 being the inner most modules
                                    top_modules=1, # the number of top modules to print aggregated profile
                                    warm_up=10, # the number of warm-ups before measuring the time of each module
                                    as_string=True, # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
                                    output_file=None, # path to the output file. If None, the profiler prints to stdout.
                                    ignore_modules=None, # the list of modules to ignore in the profiling
                                    func_name='forward') # the function name to profile, "forward" by default, for huggingface generative models, `generate` is used
```

### In Model Training Workflow

To profile model forward in a training workflow, use the `FlopsProfiler`class.
The `FlopsProfiler`class provides the following methods:
  * `start_profile()` - starts profiling
  * `get_total_flops(as_string=False)` - returns the total number of floating-point operations in the model
  * `get_total_macs(as_string=False)` - returns the total number of MACs in the model
  * `get_total_params(as_string=False)` - returns the total number of parameters in the model
  * `print_model_profile(profile_step=1, module_depth=-1, top_modules=1, detailed=True, output_file=None)` - prints the model profile
  * `stop_profile()` - stops profiling. This stops the flops counting in the model.
  * `end_profile()` - cleans up. This cleans up the profile attributes added to the model during the profiling. This should be invoked at the end of the profiling and AFTER `get_total_flops`, `get_total_macs`, `get_total_params` or `print_model_profile`.

#### Example Training Workflow

Below is an example of this usage in a typical training workflow.

```python
from flops_profiler import FlopsProfiler

model = Model()
prof = FlopsProfiler(model, ds_engine if ds_engine else None)

profile_step = 5
print_profile= True

for step, batch in enumerate(data_loader):
  # start profiling at training step "profile_step"
  if step == profile_step:
    prof.start_profile()

  # forward() method
  loss = model(batch)

  # end profiling and print output
  if step == profile_step: # if using multi nodes, check global_rank == 0 as well
    prof.stop_profile()
    flops = prof.get_total_flops()
    macs = prof.get_total_macs()
    params = prof.get_total_params()
    if print_profile:
        prof.print_model_profile(profile_step=profile_step)
    prof.end_profile()

  # runs backpropagation
  loss.backward()

  # weight update
  optimizer.step()

```

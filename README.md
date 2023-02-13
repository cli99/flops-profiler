This repo mirrors [DeepSpeed Flops Profiler](https://github.com/microsoft/DeepSpeed), can be installed by `pip install .`
# Examples
 * [bert.py](examples/bert.py)
 * [t5.py](examples/t5.py)
 * [vision.py](examples/vision.py)
 * [gpt2.py](examples/gpt2.py)
# Flops Profiler

> Measures the parameters, latency, and floating-point operations of your model.

  - [Overview](#overview)
  - [Flops Measurement](#flops-measurement)
  - [Multi-GPU, Multi-node, Data Parallelism, and Model Parallelism](#multi-gpu-multi-node-data-parallelism-and-model-parallelism)
  - [Usage](#usage)
## Overview

Effective use of hardware resources is critical to good performance, but performance inefficiency in existing implementations for large-scale model training and inference are often hard to spot and attributed to specific module components. The Flops Profiler helps users easily measure both the model training/inference speed (latency, throughput) and efficiency (floating-point operations per second, i.e., FLOPS) of a model and its submodules, with an eye towards eliminating inefficiencies in existing implementations.

Below is an example output for BERT-base on an A6000 GPU with batch size `1` and sequence length `128` (see [bert.py](examples/bert.py)):

```shell
-------------------------- DeepSpeed Flops Profiler --------------------------
Profile Summary at step 3:
Notations:
data parallel size (dp_size), model parallel size(mp_size),
number of parameters (params), number of multiply-accumulate operations(MACs),
number of floating-point operations (flops), floating-point operations per second (FLOPS),
fwd latency (forward propagation latency), bwd latency (backward propagation latency),
step (weights update latency), iter latency (sum of fwd, bwd and step latency)

params per device:                                            109.48 M
params of model = params per device * mp_size:                109.48 M
fwd MACs per device:                                          44.7 GMACs
fwd flops per device:                                         89.45 G
fwd flops of model = fwd flops per device * mp_size:          89.45 G
fwd latency:                                                  135.52 ms
fwd FLOPS per device = fwd flops per device / fwd latency:    660.07 GFLOPS

----------------------------- Aggregated Profile per device -----------------------------
Top 10 modules in terms of params, MACs or fwd latency at different model depths:
depth 0:
    params      - {'BertModel': '109.48 M'}
    MACs        - {'BertModel': '44.7 GMACs'}
    fwd latency - {'BertModel': '135.52 ms'}
depth 1:
    params      - {'BertEncoder': '85.05 M', 'BertEmbeddings': '23.84 M', 'BertPooler': '590.59 k'}
    MACs        - {'BertEncoder': '44.69 GMACs', 'BertPooler': '2.36 MMACs', 'BertEmbeddings': '0 MACs'}
    fwd latency - {'BertEncoder': '134.11 ms', 'BertEmbeddings': '1.24 ms', 'BertPooler': '173.57 us'}
depth 2:
    params      - {'ModuleList': '85.05 M', 'Embedding': '23.84 M', 'Linear': '590.59 k', 'LayerNorm': '1.54 k', 'Dropout': '0', 'Tanh': '0'}
    MACs        - {'ModuleList': '44.69 GMACs', 'Linear': '2.36 MMACs', 'Embedding': '0 MACs', 'LayerNorm': '0 MACs', 'Dropout': '0 MACs', 'Tanh': '0 MACs'}
    fwd latency - {'ModuleList': '133.94 ms', 'Embedding': '758.65 us', 'LayerNorm': '205.04 us', 'Linear': '93.94 us', 'Tanh': '37.91 us', 'Dropout': '14.78 us'}
depth 3:
    params      - {'BertLayer': '85.05 M'}
    MACs        - {'BertLayer': '44.69 GMACs'}
    fwd latency - {'BertLayer': '133.94 ms'}
depth 4:
    params      - {'BertAttention': '28.37 M', 'BertIntermediate': '28.35 M', 'BertOutput': '28.34 M'}
    MACs        - {'BertAttention': '15.7 GMACs', 'BertIntermediate': '14.5 GMACs', 'BertOutput': '14.5 GMACs'}
    fwd latency - {'BertAttention': '60.64 ms', 'BertIntermediate': '37.63 ms', 'BertOutput': '34.44 ms'}
depth 5:
    params      - {'Linear': '56.67 M', 'BertSelfAttention': '21.26 M', 'BertSelfOutput': '7.11 M', 'LayerNorm': '18.43 k', 'GELUActivation': '0', 'Dropout': '0'}
    MACs        - {'Linear': '28.99 GMACs', 'BertSelfAttention': '12.08 GMACs', 'BertSelfOutput': '3.62 GMACs', 'GELUActivation': '0 MACs', 'LayerNorm': '0 MACs', 'Dropout': '0 MACs'}
    fwd latency - {'Linear': '62.62 ms', 'BertSelfAttention': '47.87 ms', 'BertSelfOutput': '12.48 ms', 'GELUActivation': '4.41 ms', 'LayerNorm': '2.21 ms', 'Dropout': '290.63 us'}

------------------------------ Detailed Profile per device ------------------------------
Each module profile is listed after its name in the following order:
params, percentage of total params, MACs, percentage of total MACs, fwd latency, percentage of total fwd latency, fwd FLOPS

Note: 1. A module can have torch.nn.module or torch.nn.functional to compute logits (e.g. CrossEntropyLoss). They are not counted as submodules, thus not to be printed out. However they make up the difference between a parent's MACs (or latency) and the sum of its submodules'.
1. Number of floating-point operations is a theoretical estimation, thus FLOPS computed using that could be larger than the maximum system throughput.
2. The fwd latency listed in the top module's profile is directly captured at the module forward function in PyTorch, thus it's less than the fwd latency shown above which is captured in DeepSpeed.

BertModel(
  109.48 M, 100.00% Params, 44.7 GMACs, 100.00% MACs, 135.52 ms, 100.00% latency, 660.07 GFLOPS,
  (embeddings): BertEmbeddings(
    23.84 M, 21.77% Params, 0 MACs, 0.00% MACs, 1.24 ms, 0.91% latency, 1.59 GFLOPS,
    (word_embeddings): Embedding(23.44 M, 21.41% Params, 0 MACs, 0.00% MACs, 574.83 us, 0.42% latency, 0.0 FLOPS, 30522, 768, padding_idx=0)
    (position_embeddings): Embedding(393.22 k, 0.36% Params, 0 MACs, 0.00% MACs, 40.29 us, 0.03% latency, 0.0 FLOPS, 512, 768)
    (token_type_embeddings): Embedding(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 143.53 us, 0.11% latency, 0.0 FLOPS, 2, 768)
    (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 205.04 us, 0.15% latency, 9.59 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
    (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 14.78 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
  )
  (encoder): BertEncoder(
    85.05 M, 77.69% Params, 44.69 GMACs, 99.99% MACs, 134.11 ms, 98.96% latency, 666.97 GFLOPS,
    (layer): ModuleList(
      85.05 M, 77.69% Params, 44.69 GMACs, 99.99% MACs, 133.94 ms, 98.83% latency, 667.8 GFLOPS,
      (0): BertLayer(
        7.09 M, 6.47% Params, 3.72 GMACs, 8.33% MACs, 12.5 ms, 9.22% latency, 596.26 GFLOPS,
        (attention): BertAttention(
          2.36 M, 2.16% Params, 1.31 GMACs, 2.93% MACs, 5.61 ms, 4.14% latency, 466.73 GFLOPS,
          (self): BertSelfAttention(
            1.77 M, 1.62% Params, 1.01 GMACs, 2.25% MACs, 4.4 ms, 3.25% latency, 457.42 GFLOPS,
            (query): Linear(590.59 k, 0.54% Params, 301.99 MMACs, 0.68% MACs, 996.35 us, 0.74% latency, 606.19 GFLOPS, in_features=768, out_features=768, bias=True)
            (key): Linear(590.59 k, 0.54% Params, 301.99 MMACs, 0.68% MACs, 794.89 us, 0.59% latency, 759.83 GFLOPS, in_features=768, out_features=768, bias=True)
            (value): Linear(590.59 k, 0.54% Params, 301.99 MMACs, 0.68% MACs, 813.72 us, 0.60% latency, 742.24 GFLOPS, in_features=768, out_features=768, bias=True)
            (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 14.31 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            592.13 k, 0.54% Params, 301.99 MMACs, 0.68% MACs, 1.19 ms, 0.88% latency, 508.91 GFLOPS,
            (dense): Linear(590.59 k, 0.54% Params, 301.99 MMACs, 0.68% MACs, 808.72 us, 0.60% latency, 746.84 GFLOPS, in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 197.41 us, 0.15% latency, 9.96 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 10.25 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          2.36 M, 2.16% Params, 1.21 GMACs, 2.70% MACs, 3.78 ms, 2.79% latency, 639.51 GFLOPS,
          (dense): Linear(2.36 M, 2.16% Params, 1.21 GMACs, 2.70% MACs, 3.35 ms, 2.47% latency, 721.68 GFLOPS, in_features=768, out_features=3072, bias=True)
          (intermediate_act_fn): GELUActivation(0, 0.00% Params, 0 MACs, 0.00% MACs, 408.65 us, 0.30% latency, 0.0 FLOPS, )
        )
        (output): BertOutput(
          2.36 M, 2.16% Params, 1.21 GMACs, 2.70% MACs, 3.01 ms, 2.22% latency, 803.66 GFLOPS,
          (dense): Linear(2.36 M, 2.16% Params, 1.21 GMACs, 2.70% MACs, 2.62 ms, 1.93% latency, 921.53 GFLOPS, in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm(1.54 k, 0.00% Params, 0 MACs, 0.00% MACs, 191.93 us, 0.14% latency, 10.24 GFLOPS, (768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 25.51 us, 0.02% latency, 0.0 FLOPS, p=0.1, inplace=False)
        )
      )
      (1): BertLayer(...)
      ...
      (11): BertLayer(...)
    )
  )
  (pooler): BertPooler(
    590.59 k, 0.54% Params, 2.36 MMACs, 0.01% MACs, 173.57 us, 0.13% latency, 27.19 GFLOPS,
    (dense): Linear(590.59 k, 0.54% Params, 2.36 MMACs, 0.01% MACs, 93.94 us, 0.07% latency, 50.23 GFLOPS, in_features=768, out_features=768, bias=True)
    (activation): Tanh(0, 0.00% Params, 0 MACs, 0.00% MACs, 37.91 us, 0.03% latency, 0.0 FLOPS, )
  )
)
------------------------------------------------------------------------------
Number of flops:                89.45 G
Number of MACs:                 44.7 GMACs
Number of parameters:           109.48 M
```

In the summary profile, the Flops Profiler outputs the number of parameters, floating-point operations (flops), FLOPS, latency, and throughput in samples/second of the model. This profile shows how much performance gap (compared to the peak hardware performance) the current model execution has and helps users tune the training or inference setup (e.g., hyperparameters, data parallelism, model parallelism, system configurations, etc.) for better performance.

The Flops Profiler also measures significant modules at different model depths (aggregated profile) and module-specific profile in the model architecture (detailed profile). Using these profiles DeepSpeed users can understand how each layer or submodule contributes to the overall model complexity/performance. Then users can adjust or refactor the model design to achieve better performance. For example, using the profiler, DeepSpeed users can quantitatively tell if stacking smaller layers is lighter or more performant than having bigger ones. The aggregated and detailed profiles also allow users to quickly identify bottleneck modules. In the BERT-Large example above, using the Flops Profiler, we find that BertLayer is the most significant layer and contains quite a few dropout, softmax, and layer norm along with linear modules. These modules are not heavy in flops and would trigger many device kernel invocations and create excessive read/write requests to memory. The pattern shown in the detailed profile suggests this is a perfect match for kernel fusion, and we developed fused transformer-kernels to reduce data movement (See DeepSpeedBert). After applying our optimizations, we see a 25% improvement in FLOPS per device and overall training samples/second in the Flops Profiler output.

The Flops Profiler can be used with the DeepSpeed runtime without any user code change or be used independently from DeepSpeed as a standalone package. When using DeepSpeed for model training, the profiler can be enabled in the DeepSpeed configuration file. As a standalone package, the profiler API can be used in both training and inference code. The DeepSpeed profiler is still under active development and includes just initial features.  Stay connected for more exciting features to be added soon.

## Flops Measurement

Similar to existing flops calculation tools or methods, the Flops Profiler measures the flops of the forward pass of a module and the flops of the backward pass is estimated as `2` times of that of the forward pass.
Different from the PyTorch profiler which calculates the flops of PyTorch operators, the Flops Profiler measures the flops within modules in a model and provides more insights to the users about the model execution.
The flops estimation is partly inspired by [ptflops](https://github.com/sovrasov/flops-counter.pytorch) with the major difference being that the Flops Profiler not only supports flops computation directly at module level, but can also capture ```torch.nn.functional``` invoked in a module to estimate the flops.
Thus the Flops Profiler allows for customized modules in the model, e.g., ```ParallelTransformerLayerworks, ParallelSelfAttention, RowParallelLinear, etc.``` in [Megatron-LM](https://github.com/NVIDIA/Megatron-LM). This is in contrast to ptflops which requires users to write customized flops calculation functions for each customized module.

## Multi-device, Multi-node, Data Parallelism, and Model Parallelism

The Flops Profiler outputs the per device profile as well as the world size, data parallel size, and model parallel size.                                          1
For models running on multi-device or multi-node, only change of the model parallelism (e.g. ```--model-parallel-size``` in [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)) affects the number of flops and parameters profiled, i.e.,
`model_parallel_size * flops = total_flops` and `model_parallel_size * parameters = total_parameters`. The data parallel size or world size (related to the number of GPUs or nodes) does not affect the per device profile.

## Usage

The Flops Profiler can be used with the DeepSpeed runtime or as a standalone package. When using DeepSpeed for model training, the profiler can be configured in the deepspeed configuration file without user code change. To use the Flops Profiler outside of the DeepSpeed runtime, one can simply install DeepSpeed and import the flops_profiler package to use the APIs directly. Examples of each usage are given below.

- [Examples](#examples)
- [Flops Profiler](#flops-profiler)
  - [Overview](#overview)
  - [Flops Measurement](#flops-measurement)
  - [Multi-device, Multi-node, Data Parallelism, and Model Parallelism](#multi-device-multi-node-data-parallelism-and-model-parallelism)
  - [Usage](#usage)
    - [Usage Outside the DeepSpeed Runtime](#usage-outside-the-deepspeed-runtime)
      - [In Model Inference](#in-model-inference)
        - [Example: AlexNet](#example-alexnet)
      - [In Model Training Workflow](#in-model-training-workflow)
        - [Example Training Workflow](#example-training-workflow)
    - [Usage With the DeepSpeed Runtime](#usage-with-the-deepspeed-runtime)
      - [Example: Megatron-LM](#example-megatron-lm)
###  Usage Outside the DeepSpeed Runtime

The profiler can be used as a standalone package outside of the DeepSpeed runtime.
One can simply install DeepSpeed and import the `flops_profiler` package to use the APIs directly.
Refer to [installation of DeepSpeed](https://www.deepspeed.ai/getting-started/#installation) for installing DeepSpeed.

#### In Model Inference

To profile a trained model in inference, we use the `get_model_profile` function. If the inference is involed in more than just a `forward` function of the model, for example, `model.generate()`, we can use the `start_profile`, `stop_profile`, and `end_profile` to capture the higher-level function (similar to the training use case); Or pass in `mode='generate'` when calling `get_model_profile`.

Examples are given below.

##### Example: AlexNet

The following example shows how to profile AlexNet using the Flops Profiler.

```python
import torchvision.models as models
import torch
from deepspeed.profiling.flops_profiler import get_model_profile

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

#### In Model Training Workflow

To profile model forward in a training workflow, use the `FlopsProfiler`class.
The `FlopsProfiler`class provides the following methods:
  * `start_profile()` - starts profiling
  * `get_total_flops(as_string=False)` - returns the total number of floating-point operations in the model
  * `get_total_macs(as_string=False)` - returns the total number of MACs in the model
  * `get_total_params(as_string=False)` - returns the total number of parameters in the model
  * `print_model_profile(profile_step=1, module_depth=-1, top_modules=1, detailed=True, output_file=None)` - prints the model profile
  * `stop_profile()` - stops profiling. This stops the flops counting in the model.
  * `end_profile()` - cleans up. This cleans up the profile attributes added to the model during the profiling. This should be invoked at the end of the profiling and AFTER `get_total_flops`, `get_total_macs`, `get_total_params` or `print_model_profile`.

##### Example Training Workflow

Below is an example of this usage in a typical training workflow.

```python
from deepspeed.profiling.flops_profiler import FlopsProfiler

model = Model()
prof = FlopsProfiler(model)

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

### Usage With the DeepSpeed Runtime

When using DeepSpeed for model training, the profiler can be configured in the deepspeed configuration file. No explicit API calls are needed to use the profiler. The profiler can be enabled by adding the following field to the `deepspeed_config` json file. Refer to [Flops Profiler](https://www.deepspeed.ai/docs/config-json/#flops-profiler) for details.

```json
{
  "flops_profiler": {
    "enabled": true,
    "profile_step": 1,
    "module_depth": -1,
    "top_modules": 1,
    "detailed": true,
    "output_file": null
    }
}
```

#### Example: Megatron-LM

For information on running Megatron-LM with DeepSpeed, please refer to the tutorial [Megatron-LM](https://github.com/microsoft/DeepSpeedExamples/tree/master/Megatron-LM).

An example output of 12-layer Megatron-LM model (`hidden_size = 8192, num_attention_heads = 32, batch_size = 1024, seq_length = 1024`) is shown below.

```shell
-------------------------- Flops Profiler --------------------------
Profile Summary at step 10:
Notations:
data parallel size (dp_size), model parallel size(mp_size),
number of parameters (params), number of multiply-accumulate operations(MACs),
number of floating-point operations (flops), floating-point operations per second (FLOPS),
fwd latency (forward propagation latency), bwd latency (backward propagation latency),
step (weights update latency), iter latency (sum of fwd, bwd and step latency)

world size:                                                   1
data parallel size:                                           1
model parallel size:                                          1
batch size per device:                                           1024
params per gpu:                                               1.29 M
params of model = params per device * mp_size:                   1.29 M
fwd MACs per device:                                             41271.95 G
fwd flops per device:                                            82543.9 G
fwd flops of model = fwd flops per device * mp_size:             82543.9 G
fwd latency:                                                  1.89 s
bwd latency:                                                  5.38 s
fwd FLOPS per device = fwd flops per device / fwd latency:          43.68 TFLOPS
bwd FLOPS per device = 2 * fwd flops per device / bwd latency:      30.7 TFLOPS
fwd+bwd FLOPS per device = 3 * fwd flops per device / (fwd+bwd latency):   34.07 TFLOPS
step latency:                                                 34.12 s
iter latency:                                                 41.39 s
samples/second:                                               24.74

----------------------------- Aggregated Profile per device -----------------------------
Top 1 modules in terms of params, MACs or fwd latency at different model depths:
depth 0:
    params      - {'GPT2Model': '1.29 M'}
    MACs        - {'GPT2Model': '41271.95 GMACs'}
    fwd latency - {'GPT2Model': '1.84 s'}
depth 1:
    params      - {'TransformerLanguageModel': '1.29 M'}
    MACs        - {'TransformerLanguageModel': '39584.03 GMACs'}
    fwd latency - {'TransformerLanguageModel': '1.83 s'}
depth 2:
    params      - {'ParallelTransformer': '1.29 M'}
    MACs        - {'ParallelTransformer': '39584.03 GMACs'}
    fwd latency - {'ParallelTransformer': '1.81 s'}
depth 3:
    params      - {'ModuleList': '1.28 M'}
    MACs        - {'ModuleList': '39584.03 GMACs'}
    fwd latency - {'ModuleList': '1.3 s'}
depth 4:
    params      - {'ParallelTransformerLayerPart2': '688.15 k'}
    MACs        - {'ParallelTransformerLayerPart2': '26388.28 GMACs'}
    fwd latency - {'ParallelTransformerLayerPart2': '865.73 ms'}
depth 5:
    params      - {'ParallelMLP': '491.54 k'}
    MACs        - {'ParallelMLP': '26388.28 GMACs'}
    fwd latency - {'ParallelMLP': '849.4 ms'}

------------------------------ Detailed Profile per device ------------------------------
Each module profile is listed after its name in the following order:
params, percentage of total params, MACs, percentage of total MACs, fwd latency, percentage of total fwd latency, fwd FLOPS

Note: 1. A module can have torch.nn.module or torch.nn.functional to compute logits (e.g. CrossEntropyLoss). They are not counted as submodules, thus not to be printed out. However they make up the difference between a parent's MACs(or latency) and the sum of its submodules'.
2. Number of floating-point operations is a theoretical estimation, thus FLOPS computed using that could be larger than the maximum system throughput.
3. The fwd latency listed in the top module's profile is directly captured at the module forward function in PyTorch, thus it's less than the fwd latency shown above which is captured in DeepSpeed.

GPT2Model(
  1.29 M, 100.00% Params, 41271.95 GMACs, 100.00% MACs, 1.84 s, 100.00% latency, 44.78 TFLOPS,
  (language_model): TransformerLanguageModel(
    1.29 M, 100.00% Params, 39584.03 GMACs, 95.91% MACs, 1.83 s, 99.11% latency, 43.34 TFLOPS,
    (embedding): Embedding(
      2, 0.00% Params, 0 MACs, 0.00% MACs, 18.1 ms, 0.98% latency, 0.0 FLOPS,
      (word_embeddings): VocabParallelEmbedding(1, 0.00% Params, 0 MACs, 0.00% MACs, 164.75 us, 0.01% latency, 0.0 FLOPS, )
      (position_embeddings): Embedding(1, 0.00% Params, 0 MACs, 0.00% MACs, 489.23 us, 0.03% latency, 0.0 FLOPS, 1024, 8192)
      (embedding_dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 93.94 us, 0.01% latency, 0.0 FLOPS, p=0.1, inplace=False)
    )
    (transformer): ParallelTransformer(
      1.29 M, 100.00% Params, 39584.03 GMACs, 95.91% MACs, 1.81 s, 98.11% latency, 43.78 TFLOPS,
      (layers): ModuleList(
        1.28 M, 98.73% Params, 39584.03 GMACs, 95.91% MACs, 1.3 s, 70.66% latency, 60.79 TFLOPS,
        (0): ParallelTransformerLayerPart1(
          49.15 k, 3.80% Params, 1099.65 GMACs, 2.66% MACs, 23.5 ms, 1.27% latency, 93.6 TFLOPS,
          (input_layernorm): FusedLayerNorm(16.38 k, 1.27% Params, 0 MACs, 0.00% MACs, 128.75 us, 0.01% latency, 0.0 FLOPS, torch.Size([8192]), eps=1e-05, elementwise_affine=True)
          (attention): ParallelSelfAttention(
            32.77 k, 2.53% Params, 1099.65 GMACs, 2.66% MACs, 22.8 ms, 1.24% latency, 96.46 TFLOPS,
            (query_key_value): ColumnParallelLinear(24.58 k, 1.90% Params, 824.63 GMACs, 2.00% MACs, 8.93 ms, 0.48% latency, 184.7 TFLOPS, )
            (scale_mask_softmax): FusedScaleMaskSoftmax(0, 0.00% Params, 134.22 MMACs, 0.00% MACs, 151.16 us, 0.01% latency, 1.78 TFLOPS, )
            (attention_dropout): Dropout(0, 0.00% Params, 0 MACs, 0.00% MACs, 79.63 us, 0.00% latency, 0.0 FLOPS, p=0.1, inplace=False)
            (dense): RowParallelLinear(8.19 k, 0.63% Params, 274.88 GMACs, 0.67% MACs, 2.67 ms, 0.14% latency, 205.81 TFLOPS, )
          )
        )
        (1): ParallelTransformerLayerPart2(
          57.35 k, 4.43% Params, 2199.02 GMACs, 5.33% MACs, 77.53 ms, 4.21% latency, 56.73 TFLOPS,
          (post_attention_layernorm): FusedLayerNorm(16.38 k, 1.27% Params, 0 MACs, 0.00% MACs, 116.11 us, 0.01% latency, 0.0 FLOPS, torch.Size([8192]), eps=1e-05, elementwise_affine=True)
          (mlp): ParallelMLP(
            40.96 k, 3.16% Params, 2199.02 GMACs, 5.33% MACs, 76.19 ms, 4.13% latency, 57.72 TFLOPS,
            (dense_h_to_4h): ColumnParallelLinear(32.77 k, 2.53% Params, 1099.51 GMACs, 2.66% MACs, 10.79 ms, 0.59% latency, 203.81 TFLOPS, )
            (dense_4h_to_h): RowParallelLinear(8.19 k, 0.63% Params, 1099.51 GMACs, 2.66% MACs, 14.38 ms, 0.78% latency, 152.95 TFLOPS, )
          )
        )
        ...
        (23): ParallelTransformerLayerPart2(...)
      )
      (final_layernorm): FusedLayerNorm(16.38 k, 1.27% Params, 0 MACs, 0.00% MACs, 110.86 us, 0.01% latency, 0.0 FLOPS, torch.Size([8192]), eps=1e-05, elementwise_affine=True)
    )
  )
)
------------------------------------------------------------------------------
```

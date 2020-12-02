# flops-profiler

> Measures the time, number of estimated flops and parameters of each module in a PyTorch Model.

The flops-profiler profiles the forward pass of a PyTorch model and prints the model graph with the measured profile attached to each module. It shows how time, flops and parameters are spent in the model and which modules or layers could be the bottleneck. It also outputs the names of the top k modules in terms of aggregated time, flops, and parameters at depth l with k and l specified by the user. The output profile is computed for each batch of input. If multiple forward passes are specified by the user to caputre (in the case where the model have different paths or for more accurate timing), the average profile of the multiple batches is taken.

The flops estimation is partly inspired by [ptflops](https://github.com/sovrasov/flops-counter.pytorch) with the major difference being that flops-profiler captures ```torch.nn.functional``` invoked in a module to estimate the flops, thus allowing customized modules in the model (e.g. ```ParallelTransformerLayerworks, ParallelSelfAttention, RowParallelLinear, etc.``` in [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)). The flops-profiler also supports flops computation at module level (for RNNs).

For models running on multi-node or multi-gpu, only the model parallelism affects the number of flops and parameters (e.g. ```--model-parallel-size``` in [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)), i.e., model_parallel_size * flops = total_flops, model_parallel_size * parameters = total_parameters. The number of gpus or nodes does not affect the output profile.

Below is an example output for LeNet5 with batch size 1024 on a V100 GPU:
<!-- ![](header.png) -->

```
LeNet5(
  61.71 k, 100.00% Params, 439.55 MMACs, 100.00% MACs, 25.62 ms, 100.00% time, 0.034 TFLOPS,
  (feature_extractor): Sequential(
    50.69 k, 82.15% Params, 428.37 MMACs, 97.46% MACs, 18.41 ms, 71.85% time, 0.047 TFLOPS,
    (0): Conv2d(156, 0.25% Params, 125.24 MMACs, 28.49% MACs, 10.56 ms, 41.21% time, 0.024 TFLOPS, 1, 6, kernel_size=(5, 5), stride=(1, 1))
    (1): Tanh(0, 0.00% Params, 0.0 MACs, 0.00% MACs, 2.25 ms, 8.79% time, 0.0 TFLOPS, )
    (2): AvgPool2d(0, 0.00% Params, 4.82 MMACs, 1.10% MACs, 2.47 ms, 9.63% time, 0.0039 TFLOPS, kernel_size=2, stride=2, padding=0)
    (3): Conv2d(2.42 k, 3.92% Params, 247.4 MMACs, 56.28% MACs, 1.08 ms, 4.23% time, 0.46 TFLOPS, 6, 16, kernel_size=(5, 5), stride=(1, 1))
    (4): Tanh(0, 0.00% Params, 0.0 MACs, 0.00% MACs, 497.39 us, 1.94% time, 0.0 TFLOPS, )
    (5): AvgPool2d(0, 0.00% Params, 1.64 MMACs, 0.37% MACs, 758.24 us, 2.96% time, 0.0043 TFLOPS, kernel_size=2, stride=2, padding=0)
    (6): Conv2d(48.12 k, 77.98% Params, 49.27 MMACs, 11.21% MACs, 606.35 us, 2.37% time, 0.16 TFLOPS, 16, 120, kernel_size=(5, 5), stride=(1, 1))
    (7): Tanh(0, 0.00% Params, 0.0 MACs, 0.00% MACs, 68.86 us, 0.27% time, 0.0 TFLOPS, )
  )
  (classifier): Sequential(
    11.01 k, 17.85% Params, 11.18 MMACs, 2.54% MACs, 7.03 ms, 27.43% time, 0.0032 TFLOPS,
    (0): Linear(10.16 k, 16.47% Params, 10.32 MMACs, 2.35% MACs, 2.71 ms, 10.57% time, 0.0076 TFLOPS, in_features=120, out_features=84, bias=True)
    (1): Tanh(0, 0.00% Params, 0.0 MACs, 0.00% MACs, 78.77 us, 0.31% time, 0.0 TFLOPS, )
    (2): Linear(850, 1.38% Params, 860.16 KMACs, 0.20% MACs, 4.17 ms, 16.27% time, 0.00041 TFLOPS, in_features=84, out_features=10, bias=True)
  )
)
Top 3 modules in flops at depth 2 are {'Conv2d': '421.91 MMACs', 'Linear': '11.18 MMACs', 'AvgPool2d': '6.46 MMACs'}
Top 3 modules in params at depth 2 are {'Conv2d': '50.69 k', 'Linear': '11.01 k', 'Tanh': '0'}
Top 3 modules in time at depth 2 are {'Conv2d': '12.25 ms', 'Linear': '6.88 ms', 'AvgPool2d': '3.23 ms'}
Batch size:                     1024
Number of multiply-adds:        439.55 MMACs
Number of parameters:           61.71 k
Number of steps profiled:       10
```

## Installation


```
python -m pip install --upgrade git+https://github.com/cli99/flops-profiler.git
```


## Usage
### Use the high level-API and run the model inference for profiling purpose

```python
import torchvision.models as models
import torch
from flops_profiler import get_model_profile

with torch.cuda.device(0):
    mod = models.alexnet()
    batch_size = 256
    macs, params, steps = get_model_profile(model, # the PyTorch model to be profiled
                                     input_res=(batch_size, 3, 224, 224), # input shape or input to the input_constructor
                                     input_constructor=None, # If specified, the constructor is applied to input_res and the constructor output is used as the input to the model
                                     print_profile=True, # whether to print the model graph with the profile annotated. Defaults to True
                                     print_aggregated_profile=True, # whether to print the aggregated profile for top modules. Defaults to True
                                     module_depth=-1, # the depth into the nested modules. Defaults to -1 (the inner most modules)
                                     top_modules=3, # the number of top modules to print aggregated profile
                                     warm_up=10, # the number of warm-up steps before measuring the time of each module. Defaults to 5
                                     num_steps=10, # the number of steps to profile. Defaults to 10
                                     as_strings=True, # whether to print the output as strings (e.g. 1k). Defaults to True
                                     ignore_modules=None) # the list of modules to ignore during profiling. Defaults to None

    print("{:<30}  {:<8}".format("Batch size: ", batch_size))
    print('{:<30}  {:<8}'.format('Number of multiply-adds: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    print("{:<30}  {:<8}".format("Number of steps profiled: ", steps))
```

Examples of this usage is given in [examples](examples).

### Use the low-level APIs to profile the forward pass in the existing model training workflow

- `start_profile` - starts profiling
- `get_total_flops` - returns the total number of flops
- `get_total_params` - returns the total number of params
- `get_total_duration` - returns the total duration of the model forward pass
- `get_total_steps` - returns the total number of steps (or input batches) profiled.
- `print_model_profile` - prints the profile annotated
- `print_model_aggregated_profile` - prints the aggregated profile for the top modules
- `end_profile` - ends profiling and cleans up, invoked at the end of the profiling and before any printing method.

`flops_to_string`, `params_to_string`, `duration_to_string` are utility functions to convert the metric number to string.

Below is an example of this usage in a typical training workflow.

```python
from flops_profiler.profiler import FlopsProfiler

model = Model()
profiler = FlopsProfiler(model)

start_step = 5
end_step = 10
assert (end_step > start_step), "should end profiling after start profiling"
print_profile = True
pring_aggregated_profile = True

for step, batch in enumerate(data_loader):
  # start profiling at training step "profile_step"
  if step == start_step:
    profiler.start_profile()

  # end profiling and print output at training step "profile_step"
  if model == end_step: # if using multi nodes, check global_rank == 0 as well
    flops = profiler.get_total_flops()
    params = profiler.get_total_flops()
    duration = profiler.get_total_duration()
    steps = profiler.get_total_steps()
    if print_profile:
        profiler.print_model_profile()
    if print_aggregated_profile:
        profiler.print_model_aggregated_profile(module_depth=-1, top_modules=3)
    profiler.end_profile()
    print(flops, params, duration, step)

  # forward() method
  loss = model(batch)

  # runs backpropagation
  loss.backward()

  # weight update
  optimizer.step()
```

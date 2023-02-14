import pytest
import torch

from flops_profiler.profiler import get_model_profile, FlopsProfiler
from tests.lenet5 import LeNet5_10

def test_get_model_profile(LeNet5_10):
    batch_size = 1024
    input = torch.randn(batch_size, 1, 32, 32)
    flops, macs, params = get_model_profile(
        LeNet5_10,
        tuple(input.shape),
    )
    assert macs == 426516480
    assert params == 61706

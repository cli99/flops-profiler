import pytest
import torch
import torch.nn as nn
import os
import tempfile

from flops_profiler.profiler import get_model_profile, FlopsProfiler
from tests.lenet5 import LeNet5_10


def test_get_model_profile(LeNet5_10, tmpdir):
    batch_size = 1024
    input = torch.randn(batch_size, 1, 32, 32)
    flops, macs, params = get_model_profile(
        LeNet5_10,
        tuple(input.shape),
        output_file=tmpdir.join('tmp.txt'),
    )
    assert macs == 426516480
    assert params == 61706

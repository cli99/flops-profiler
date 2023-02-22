import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from flops_profiler.profiler import get_model_profile, FlopsProfiler
from tests.lenet5 import LeNet5_10

class Engine:
    def __init__(self, world_size, dp_world_size, mp_world_size):
        self.world_size = world_size
        self.dp_world_size = dp_world_size # does not affect profile
        self.mp_world_size = mp_world_size
    def train_micro_batch_size_per_gpu(self):
        return 8

def test_multi_devices(LeNet5_10, tmpdir):
    batch_size = 1024
    input = torch.randn(batch_size, 1, 32, 32)
    engine = Engine(1, 1, 1)
    prof = FlopsProfiler(LeNet5_10, engine)

    prof.start_profile()
    LeNet5_10(input)
    prof.stop_profile()
    params = prof.get_total_params()
    macs = prof.get_total_macs()
    prof.print_model_profile()
    prof.end_profile()
    assert macs == 426516480
    assert params == 61706

import pytest
import torch

from flops_profiler.profiler import get_model_profile


class LeNet5(torch.nn.Module):

    def __init__(self, n_classes):
        super(LeNet5, self).__init__()

        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=6,
                            kernel_size=5,
                            stride=1),
            torch.nn.Tanh(),
            torch.nn.AvgPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=6,
                            out_channels=16,
                            kernel_size=5,
                            stride=1),
            torch.nn.Tanh(),
            torch.nn.AvgPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=16,
                            out_channels=120,
                            kernel_size=5,
                            stride=1),
            torch.nn.Tanh(),
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=120, out_features=84),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=84, out_features=n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = torch.nn.functional.softmax(logits, dim=1)
        return logits, probs


@pytest.fixture
def lenet5_10():
    return LeNet5(10)


def test_flops_profiler_in_inference(lenet5_10):
    batch_size = 1024
    input = torch.randn(batch_size, 1, 32, 32)
    flops, macs, params = get_model_profile(
        lenet5_10,
        tuple(input.shape),
        print_profile=False,
        as_string=False,
    )
    assert macs == 426516480
    assert params == 61706
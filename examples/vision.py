import argparse
import torch
import torchvision.models as models
from flops_profiler.profiler import get_model_profile
import utils

pt_models = {
    'resnet18': models.resnet18,
    'resnet50': models.resnet50,
    'alexnet': models.alexnet,
    'vgg16': models.vgg16,
    'squeezenet': models.squeezenet1_0,
    'densenet': models.densenet161,
    'inception': models.inception_v3
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='flops-profiler example script')
    parser.add_argument(
        '--cuda-device',
        type=int,
        default=0,
        help='Cuda device to run the model if available, else cpu is used.')
    parser.add_argument('--model',
                        choices=list(pt_models.keys()),
                        type=str,
                        default='resnet50')
    args = parser.parse_args()

    model = pt_models[args.model]()
    use_cuda=True
    device = torch.device('cuda:0') if torch.cuda.is_available() and use_cuda else torch.device('cpu')
    model = model.to(device)

    batch_size = 1
    flops, macs, params = get_model_profile(model, (batch_size, 3, 224, 224),
                                            print_profile=True,
                                            module_depth=-1,
                                            top_modules=3,
                                            warm_up=5,
                                            as_string=True,
                                            ignore_modules=None)

    utils.print_output(flops, macs, params)
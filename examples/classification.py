import argparse
import torch
import torchvision.models as models
from flops_profiler.profiler import get_model_profile

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
    parser.add_argument('--device',
                        type=int,
                        default=0,
                        help='Device to store the model.')
    parser.add_argument('--model',
                        choices=list(pt_models.keys()),
                        type=str,
                        default='resnet18')
    args = parser.parse_args()

    net = pt_models[args.model]()

    if torch.cuda.is_available():
        net.cuda(device=args.device)

    batch_size = 256
    macs, params = get_model_profile(net, (batch_size, 3, 224, 224),
                                     print_profile=True,
                                     print_aggregated_profile=True,
                                     module_depth=-1,
                                     top_modules=3,
                                     warm_up=5,
                                     as_string=True,
                                     ignore_modules=None)

    print("{:<30}  {:<8}".format("Batch size: ", batch_size))
    print('{:<30}  {:<8}'.format('Number of MACs: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

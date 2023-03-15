# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import onnx
import torch
from onnxsim import simplify

from mmpose.apis import init_model


def _convert_batchnorm(module):
    """Convert the syncBNs into normal BN3ds."""
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm2d(module.num_features, module.eps,
                                             module.momentum, module.affine,
                                             module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep requires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, _convert_batchnorm(child))
    del module
    return module_output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('save_path', help='onnx save path')
    parser.add_argument(
        '--input_size',
        type=int,
        nargs=2,
        default=[192, 256],
        help='network input size')
    parser.add_argument('--opset', type=int, default=11, help='opset version')
    args = parser.parse_args()
    return args


def export(args):
    model = init_model(args.config, args.checkpoint, device='cpu')
    model = _convert_batchnorm(model)
    model.head.forward = model.head.forward_with_decode
    dummy_image = torch.zeros((1, 3, *args.input_size[::-1]), device='cpu')

    torch.onnx.export(
        model,
        dummy_image,
        args.save_path,
        input_names=['input'],
        dynamic_axes={'input': {
            0: 'batch'
        }})

    onnx_model = onnx.load(args.save_path)
    onnx_model_simp, check = simplify(onnx_model)
    assert check, 'Simplified ONNX model could not be validated'
    onnx.save(onnx_model_simp, args.save_path)


if __name__ == '__main__':
    args = parse_args()
    export(args)

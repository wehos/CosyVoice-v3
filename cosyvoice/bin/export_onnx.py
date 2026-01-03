# Copyright (c) 2024 Antgroup Inc (authors: Zhoubofan, hexisyztem@icloud.com)
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import argparse
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import os
import sys
import torch
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../..'.format(ROOT_DIR))
sys.path.append('{}/../../third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import logging


def get_dummy_input(batch_size, seq_len, out_channels, device):
    x = torch.rand((batch_size, out_channels, seq_len), dtype=torch.float32, device=device)
    mask = torch.ones((batch_size, 1, seq_len), dtype=torch.float32, device=device)
    mu = torch.rand((batch_size, out_channels, seq_len), dtype=torch.float32, device=device)
    t = torch.rand((batch_size), dtype=torch.float32, device=device)
    spks = torch.rand((batch_size, out_channels), dtype=torch.float32, device=device)
    cond = torch.rand((batch_size, out_channels, seq_len), dtype=torch.float32, device=device)
    return x, mask, mu, t, spks, cond


def get_args():
    parser = argparse.ArgumentParser(description='export your model for deployment')
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice-300M',
                        help='local path')
    args = parser.parse_args()
    print(args)
    return args


def get_dummy_input_fp16(batch_size, seq_len, out_channels, device):
    x = torch.rand((batch_size, out_channels, seq_len), dtype=torch.float16, device=device)
    mask = torch.ones((batch_size, 1, seq_len), dtype=torch.float16, device=device)
    mu = torch.rand((batch_size, out_channels, seq_len), dtype=torch.float16, device=device)
    t = torch.rand((batch_size), dtype=torch.float16, device=device)
    spks = torch.rand((batch_size, out_channels), dtype=torch.float16, device=device)
    cond = torch.rand((batch_size, out_channels, seq_len), dtype=torch.float16, device=device)
    return x, mask, mu, t, spks, cond


@torch.no_grad()
def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')

    model = AutoModel(model_dir=args.model_dir)

    # export flow decoder estimator
    estimator = model.model.flow.decoder.estimator
    estimator.eval()
    device = model.model.device
    batch_size, seq_len = 2, 256
    out_channels = model.model.flow.decoder.estimator.out_channels

    # FP32 export
    x, mask, mu, t, spks, cond = get_dummy_input(batch_size, seq_len, out_channels, device)
    torch.onnx.export(
        estimator,
        (x, mask, mu, t, spks, cond),
        '{}/flow.decoder.estimator.fp32.onnx'.format(args.model_dir),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['x', 'mask', 'mu', 't', 'spks', 'cond'],
        output_names=['estimator_out'],
        dynamic_axes={
            'x': {2: 'seq_len'},
            'mask': {2: 'seq_len'},
            'mu': {2: 'seq_len'},
            'cond': {2: 'seq_len'},
            'estimator_out': {2: 'seq_len'},
        },
        dynamo=False,
    )
    logging.info('Exported FP32 ONNX to {}/flow.decoder.estimator.fp32.onnx'.format(args.model_dir))

    # FP16 export
    estimator_fp16 = estimator.half()
    x, mask, mu, t, spks, cond = get_dummy_input_fp16(batch_size, seq_len, out_channels, device)
    torch.onnx.export(
        estimator_fp16,
        (x, mask, mu, t, spks, cond),
        '{}/flow.decoder.estimator.fp16.onnx'.format(args.model_dir),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['x', 'mask', 'mu', 't', 'spks', 'cond'],
        output_names=['estimator_out'],
        dynamic_axes={
            'x': {2: 'seq_len'},
            'mask': {2: 'seq_len'},
            'mu': {2: 'seq_len'},
            'cond': {2: 'seq_len'},
            'estimator_out': {2: 'seq_len'},
        },
        dynamo=False,
    )
    logging.info('Exported FP16 ONNX to {}/flow.decoder.estimator.fp16.onnx'.format(args.model_dir))
    logging.info('Use load_trt() for TensorRT conversion and inference.')


if __name__ == "__main__":
    main()

from __future__ import annotations

from functools import partial

import torch
import utils
from transformers import AutoConfig
from transformers import AutoModel
from transformers import AutoTokenizer

from flops_profiler.profiler import get_model_profile


def bert_input_constructor(batch_size, seq_len, tokenizer, device):
    fake_seq = ''
    # ignore the two special tokens [CLS] and [SEP]
    for _ in range(seq_len - 2):
        fake_seq += tokenizer.pad_token
    inputs = tokenizer(
        [fake_seq] * batch_size,
        padding=True,
        truncation=True,
        return_tensors='pt',
    ).to(device)
    inputs = dict(inputs)
    return inputs


name = 'bert-base-uncased'
use_cuda = True
device = torch.device('cuda:0') if torch.cuda.is_available(
) and use_cuda else torch.device('cpu')
tokenizer = AutoTokenizer.from_pretrained(name)
config = AutoConfig.from_pretrained(name)
model = AutoModel.from_config(config)
# model.encoder.layer = model.encoder.layer[:1]
model = model.to(device)

batch_size = 1
seq_len = 128

flops, macs, params = get_model_profile(
    model,
    kwargs=bert_input_constructor(batch_size, seq_len, tokenizer, device),
    print_profile=True,
    detailed=True,
    warm_up=0,
)

utils.print_output(flops, macs, params)

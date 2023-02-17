from __future__ import annotations

import torch
import utils
from transformers import AutoConfig, AutoModel, GPT2Model, GPT2LMHeadModel, GPT2Tokenizer

from flops_profiler.profiler import get_model_profile

name = 'gpt2'
use_cuda = True
device = torch.device('cuda:0') if torch.cuda.is_available(
) and use_cuda else torch.device('cpu')
config = GPT2Tokenizer.from_pretrained(name)
model = GPT2LMHeadModel.from_pretrained(name)
model = model.to(device)

batch_size = 1
seq_len = 128
input = utils.create_test_tokens(batch_size, seq_len, device=device)

flops, macs, params = get_model_profile(
    model,
    kwargs=input,
    print_profile=True,
    detailed=True,
)

utils.print_output(flops, macs, params)

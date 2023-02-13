from functools import partial

import torch
from flops_profiler.profiler import get_model_profile
from transformers import AutoConfig, AutoModel, AutoTokenizer
import utils


def bert_input_constructor(batch_size, seq_len, tokenizer, device):
    fake_seq = ""
    for _ in range(seq_len - 2):  # ignore the two special tokens [CLS] and [SEP]
      fake_seq += tokenizer.pad_token
    inputs = tokenizer([fake_seq] * batch_size,
                       padding=True,
                       truncation=True,
                       return_tensors="pt").to(device)
    inputs = dict(inputs)
    return inputs

name = 'bert-base-uncased'
use_cuda = True
device = torch.device('cuda:0') if torch.cuda.is_available() and use_cuda else torch.device('cpu')
tokenizer = AutoTokenizer.from_pretrained(name)
config = AutoConfig.from_pretrained(name)
model = AutoModel.from_config(config)
model = model.to(device)

batch_size = 1
seq_len = 128

flops, macs, params = get_model_profile(
    model,
    kwargs=bert_input_constructor(batch_size, seq_len, tokenizer, device),
    print_profile=True,
    detailed=True,
)

utils.print_output(flops, macs, params)

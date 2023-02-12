import torch
from transformers import AutoConfig, AutoModel
from transformers import GPT2Tokenizer, GPT2Model
import utils
from flops_profiler.profiler import get_model_profile

name = "gpt2"
config = AutoConfig.from_pretrained(name)
model = AutoModel.from_config(config)

batch_size = 1
seq_len = 128
input = utils.create_test_tokens(batch_size, seq_len)

flops, macs, params = get_model_profile(
    model,
    kwargs=input,
    print_profile=True,
    detailed=True,
)

utils.print_output(flops, macs, params)

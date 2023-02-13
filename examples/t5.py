import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from flops_profiler.profiler import FlopsProfiler, get_model_profile
import utils

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")
use_cuda=True
device = torch.device('cuda:0') if torch.cuda.is_available() and use_cuda else torch.device('cpu')
model = model.to(device)

batch_size = 1
seq_len = 128
input_ids = torch.randint(
    low=1000, high=10000, size=(batch_size, seq_len), dtype=torch.int64, device=device
)

flops, macs, params = get_model_profile(
    model,
    args=[input_ids],
    print_profile=True,
    detailed=True,
    module_depth=-1,
    top_modules=1,
    func_name="generate",
)

utils.print_output(flops, macs, params)

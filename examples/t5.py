import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from flops_profiler.profiler import FlopsProfiler, get_model_profile

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

input_ids = tokenizer("translate English to German: The house is wonderful.",
                      return_tensors="pt").input_ids

model.eval()

flops, macs, params = get_model_profile(
    model,
    args=[input_ids],
    print_profile=True,
    detailed=True,
    warm_up=0,
    module_depth=-1,
    top_modules=1,
    mode='generate',
)

print("{:<30}  {:<8}".format("Number of flops: ", flops))
print("{:<30}  {:<8}".format("Number of MACs: ", macs))
print("{:<30}  {:<8}".format("Number of parameters: ", params))

# Number of flops:                7.29 G
# Number of MACs:                 3.64 GMACs
# Number of parameters:           60.51 M
# chengli in lambda-dual in flops-profiler/e
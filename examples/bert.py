from functools import partial

import torch
from transformers import BertForSequenceClassification, BertTokenizer
from flops_profiler.profiler import get_model_profile


def bert_input_constructor(batch_size, seq_len, tokenizer):
    fake_seq = ""
    for _ in range(seq_len - 2):  # ignore the two special tokens [CLS] and [SEP]
      fake_seq += tokenizer.pad_token
    inputs = tokenizer([fake_seq] * batch_size,
                       padding=True,
                       truncation=True,
                       return_tensors="pt")
    labels = torch.tensor([1] * batch_size)
    inputs = dict(inputs)
    inputs.update({"labels": labels})
    return inputs


if __name__ == '__main__':
    with torch.cuda.device(0):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        batch_size = 4
        seq_len = 128
        flops, macs, params = get_model_profile(
            model,
            kwargs=bert_input_constructor(batch_size, seq_len, tokenizer),
            print_profile=True,
            detailed=True,
        )
        print("{:<30}  {:<8}".format("Number of flops: ", flops))
        print("{:<30}  {:<8}".format("Number of MACs: ", macs))
        print("{:<30}  {:<8}".format("Number of parameters: ", params))

# Output:
# Number of multiply-adds:        21.74 GMACs
# Number of parameters:           109.48 M

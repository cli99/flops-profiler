from __future__ import annotations

import time
from typing import Any
from typing import Callable
from typing import cast
from typing import DefaultDict
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TypeVar
from typing import Union

import torch


def create_test_tokens(
    batch_size, seq_len, device=torch.device('cpu'), has_token_type_ids=False,
):
    tokens = {}
    mask_lens = torch.randint(low=1, high=seq_len, size=(batch_size,))

    mask = (
        torch.arange(seq_len).expand(batch_size, seq_len)
        < mask_lens.unsqueeze(1)
    ).to(device)
    tokens['attention_mask'] = mask.to(torch.int64)

    tokens['input_ids'] = torch.randint(
        low=1000, high=10000, size=(batch_size, seq_len), dtype=torch.int64,
    ).to(device)

    tokens['input_ids'] = tokens['input_ids'].masked_fill(~mask, 0)
    for i in range(batch_size):
        tokens['input_ids'][i, 0] = 101
        tokens['input_ids'][i, mask_lens[i] - 1] = 102

    if has_token_type_ids:
        tokens['token_type_ids'] = torch.LongTensor(
            [[0] * seq_len] * batch_size,
        ).to(device)
    return tokens


def print_output(flops, macs, params):
    print('{:<30}  {:<8}'.format('Number of flops: ', flops))
    print('{:<30}  {:<8}'.format('Number of MACs: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

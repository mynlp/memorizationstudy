import numpy as np
import pandas as pd
import torch

def indices_check():
    pass

def read_csv(addr):
    csv_sheet = pd.read_csv(addr)
    return csv_sheet


def embedding_obtain(dataset, model, idx_list, context_size, continuation_size):
    batched_context_tokens = []
    batched_true_continuation = []
    for idx in idx_list:
        data = dataset[idx]
        context_tokens = data[:context_size].tolist()
        true_continuation = data[context_size:context_size+continuation_size].tolist()
        batched_context_tokens.append(context_tokens)
        batched_true_continuation.append(true_continuation)
    if torch.cuda.is_available():
        context_tokens = torch.tensor(batched_context_tokens).to('cuda')
        true_continuation = torch.tensor(batched_true_continuation).to('cuda')
    else:
        context_tokens = torch.tensor(batched_context_tokens)
        true_continuation = torch.tensor(batched_true_continuation)
    generations = model.generate(context_tokens, temperature=0.0, top_k=0, top_p=0, max_length=context_size+continuation_size, min_length=context_size+continuation_size)
    accuracies = (true_continuation == generations[0][:, context_size:context_size+continuation_size]).float().mean(axis=-1)
    return [generations, accuracies]




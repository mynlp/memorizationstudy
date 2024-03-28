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
    try:
        generations = model.generate(context_tokens, temperature=0.0, top_k=0, top_p=0, max_length=context_size+continuation_size, min_length=context_size+continuation_size)
        accuracies = (true_continuation == generations[0][:, context_size:context_size + continuation_size]).float().mean(axis=-1)
        return [generations, accuracies]
    except torch.cuda.OutOfMemoryError:
        generations = model.generate(context_tokens[:250], temperature=0.0, top_k=0, top_p=0, max_length=context_size+continuation_size, min_length=context_size+continuation_size)
        generations1 = model.generate(context_tokens[250:], temperature=0.0, top_k=0, top_p=0, max_length=context_size+continuation_size, min_length=context_size+continuation_size)
        predictions=torch.concat((torch.generations[0][:, context_size:context_size + continuation_size],
                      torch.generations[1][:, context_size:context_size + continuation_size]), dim=0)
        accuracies = (true_continuation == predictions).float().mean(axis=-1)
        results = []
        idx = 0
        for a, b in zip(generations.hidden_states, generations1.hidden_states):
            results.append([])
            for sub_a, sub_b in zip(a, b):
                results[idx].append(torch.cat((sub_a, sub_b), dim=0))
            idx += 1
        generations.hidden_states = results
        return [generations, accuracies]

def logits_obtain(dataset, model, idx_list, context_size, continuation_size):
    batched_context_tokens = []
    batched_true_continuation = []
    for idx in idx_list:
        data = dataset[idx]
        context_tokens = data[:context_size].tolist()
        true_continuation = data[context_size:context_size + continuation_size].tolist()
        batched_context_tokens.append(context_tokens)
        batched_true_continuation.append(true_continuation)

    model_outputs = model.generate(context_tokens, temperature=0.0, top_k=0, top_p=0, max_length=context_size+continuation_size, min_length=context_size+continuation_size)
    logits = model_outputs["scores"]
    highest_probability_at_idx = []
    for idx in range(continuation_size):
        probability_scores = torch.nn.functional.softmax(logits[idx], dim=1)
        highest_probability_at_idx.append(probability_scores.max(1)[0])
    return highest_probability_at_idx






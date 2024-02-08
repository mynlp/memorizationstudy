import os
import logging
import time
import datetime
import torch
import torch.distributed as dist
import transformers.utils as transformer_utils
import multiprocessing as mp
from pythia.utils.mmap_dataset import MMapIndexedDataset
from transformers import GPTNeoXForCausalLM, AutoTokenizer


model_name = "EleutherAI/pythia-70m-deduped-v0"
CHECKPOINT= 143000
model = GPTNeoXForCausalLM.from_pretrained(
    model_name,
    revision=f'step{CHECKPOINT}',
).half().eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/pythia-70m-deduped-v0",
    revision=f'step143000',
    cache_dir="./pythia-70m-deduped-v0/step143000",
)
prefix = 'undeduped_merge/document.bin'
if "deduped" in model_name:
    prefix = 'deduped_merge/document.bin'
print(prefix)
buff_size = 2049*1024*2
print("Building dataset")
mmap_ds = MMapIndexedDataset(prefix, skip_warmup=True)
context_tokens = []
true_continuation = []
i = 0
total_num_sequences = CHECKPOINT * 1024
NUM_PROCS = 1
RANK = 0
num_sequences_per_proc = total_num_sequences // NUM_PROCS
start_idx = num_sequences_per_proc * RANK
end_idx = num_sequences_per_proc * (RANK + 1) - 1
start_idx = num_sequences_per_proc * RANK
end_idx = num_sequences_per_proc * (RANK + 1) - 1
if RANK == (NUM_PROCS - 1):
    end_idx = total_num_sequences - 1
BATCH_SIZE = 1024


for i in range(start_idx, end_idx + 1, BATCH_SIZE):
    data = mmap_ds[i:i + BATCH_SIZE]
    context_tokens.extend(data[:, :32].tolist())
    true_continuation.extend(data[:, 32:64].tolist())
    i += len(context_tokens)
    with torch.no_grad():
        context_tokens = torch.tensor(context_tokens).to('cuda')
        true_continuation = torch.tensor(true_continuation).to('cuda')
        generations = model.generate(context_tokens, temperature = 0.0, top_k = 0, top_p = 0, max_length = 64, min_length = 64)
        accuracies = (true_continuation == generations[:,32:64]).float().mean(axis=-1)
    print(f"The Contentxt:{tokenizer.batch_decode(context_tokens)}")
    print(f"The True Continuation:{tokenizer.batch_decode(true_continuation)}")
    print(f"The Generated Text:{tokenizer.batch_decode(generations)}")
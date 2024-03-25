import pandas as pd
from pythia.utils.mmap_dataset import MMapIndexedDataset
import torch
import random
from tqdm import tqdm
prefix = 'deduped_merge/document.bin'
results_70 = pd.read_csv("generate_results/memorization_evals_70m-deduped-v0_32_48_143000.csv", index_col=0)
results_160 = pd.read_csv("generate_results/memorization_evals_160m-deduped-v0_32_48_143000.csv", index_col=0)
results_410 = pd.read_csv("generate_results/memorization_evals_410m-deduped-v0_32_48_143000.csv", index_col=0)
results_1b = pd.read_csv("generate_results/memorization_evals_1b-deduped-v0_32_48_143000.csv", index_col=0)

memorized_results_70 = results_70[results_70['score'] == 1]
memorized_results_160 = results_160[results_160['score'] == 1]
memorized_results_410 = results_410[results_410['score'] == 1]
memorized_results_1b = results_1b[results_1b['score'] == 1]
unmemorized = results_1b[results_1b['score'] == 0]

idx_70 = set(memorized_results_70["idx"].tolist())
idx_160 = set(memorized_results_160["idx"].tolist())
idx_410 = set(memorized_results_410["idx"].tolist())
idx_1b = set(memorized_results_1b["idx"].tolist())
unmemorized_idx = set(unmemorized["idx"].tolist())
mmap_ds = MMapIndexedDataset(prefix, skip_warmup=True)
cross_all = idx_70.intersection(idx_160, idx_410, idx_1b)

# context_tokens = []
# for i in tqdm(list(cross_all)):
#     data = mmap_ds[i]
#     context_tokens.extend(data.tolist())
#     i += len(context_tokens)
# context_tokens = torch.tensor(context_tokens)
# torch.save(context_tokens, "cross_remembered/context_tokens.pt")

context_tokens = []
unmemorized = random.sample(unmemorized_idx, len(cross_all)*3)
for i in tqdm(list(unmemorized)):
    data = mmap_ds[i]
    context_tokens.extend(data.tolist())
    i += len(context_tokens)
context_tokens = torch.tensor(context_tokens)
torch.save(context_tokens, "cross_remembered/unmemorized.pt")


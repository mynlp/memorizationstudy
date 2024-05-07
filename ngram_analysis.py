import pandas
import json
from pythia.utils.mmap_dataset import MMapIndexedDataset
from tqdm import tqdm
import ijson
import pandas as pd
import random
import torch

sizes = ["70m", "160m", "410m", "1b", "2.8b", "6.9b", "12b"]
context_length = ["32"]
target_length = ["48"]
num_points = 10000

memorized, half_memorized, forgotten = dict(), dict(), dict()
index_set = set()
with open("/work/gk77/share/memorizationstudy_freq/tokenid_frequency.1gram.json") as file:
    n_gram_dict = json.load(file)
def read_by_idx(mmap_ds, idx_list):
    by_index_frequency_batched = []
    for idx in idx_list:
        sent_tokens = mmap_ds[int(idx)]
        by_index_frequency = []
        for token in sent_tokens:
            by_index_frequency.append(n_gram_dict[token])
        by_index_frequency_batched.append(by_index_frequency)
    return by_index_frequency_batched

mmap_ds = MMapIndexedDataset('deduped_merge/document.bin', skip_warmup=True)
for size in sizes:
    df = pd.read_csv(f"generate_results/memorization_evals_{size}-deduped-v0_32_48_143000.csv", index_col=0)
    memorized = df[df['score'] == 1]
    half_memorized = df[df['score'] == 0.5]
    forgotten = df[df['score'] == 0]
    idx_full_memorization = memorized.tolist()
    idx_not_full_memorization = half_memorized.tolist()
    idx_half_memorization = forgotten.tolist()
    memorized_index = random.sample(idx_full_memorization, num_points)
    half_memorized_index = random.sample(idx_not_full_memorization, num_points)
    forgotten_index = random.sample(idx_half_memorization, num_points)
    memorized_batched = read_by_idx(mmap_ds, memorized_index)
    half_memorized_batched = read_by_idx(mmap_ds, half_memorized_index)
    forgotten_batched = read_by_idx(mmap_ds, forgotten_index)
    averaged_memorized = torch.tensor(memorized_batched).mean(1)
    averaged_half_memorized = torch.tensor(half_memorized_batched).mean(1)
    averaged_forgotten = torch.tensor(forgotten_batched).mean(1)
    print(averaged_memorized[:48])
    print(averaged_half_memorized[:48])
    print(averaged_forgotten[:48])
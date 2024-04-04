import pandas as pd
from pythia.utils.mmap_dataset import MMapIndexedDataset
import torch
import random
from tqdm import tqdm
from transformers import AutoTokenizer
import argparse

random.seed(42)

model_name = "EleutherAI/pythia-160m-deduped-v0"
CHECKPOINT = 143000
tokenizer = AutoTokenizer.from_pretrained(
  model_name,
  revision=f"step{CHECKPOINT}",
  cache_dir=f"./pythia-160m-deduped/step{CHECKPOINT}",
)
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
paser = argparse.ArgumentParser()
paser.add_argument("--distribution_idx", type=int, default=0)
args = paser.parse_args()

idx_70 = set(memorized_results_70["idx"].tolist())
idx_160 = set(memorized_results_160["idx"].tolist())
idx_410 = set(memorized_results_410["idx"].tolist())
idx_1b = set(memorized_results_1b["idx"].tolist())
unmemorized_idx = unmemorized["idx"].tolist()
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
idx = 0
# 举例，这是你不想包括的idx
unmemorized = random.sample(unmemorized_idx, len(cross_all)*10)
part_size = len(unmemorized) // 5

# 创建3个子列表
part1 = unmemorized[0:part_size]
part2 = unmemorized[part_size:1*part_size]
part3 = unmemorized[2*part_size:3*part_size]
part4 = unmemorized[3*part_size:4*part_size]
part5 = unmemorized[4*part_size:]


# 第2步: 创建可用于抽样的索引列表
# if args.distribution_idx == 0:
#     excluded_idx = part1
# elif args.distribution_idx == 1:
#     excluded_idx = part2
# elif args.distribution_idx == 2:
#     excluded_idx = part3
# elif args.distribution_idx == 3:
#     excluded_idx = part4
# elif args.distribution_idx == 4:
#     excluded_idx = part5
#
# available_idx = [i for i in unmemorized_idx if i not in set(excluded_idx)]
for i in tqdm(part1):
    data = mmap_ds[i]
    idx += 1
    text = tokenizer.decode(data)
    context_tokens.append([idx, text])
df = pd.DataFrame(context_tokens, columns=["idx", "text"])
df.to_json(f"cross_remembered/unmemorized_text_{args.distribution_idx}.json", index=False, orient='records', lines=True)




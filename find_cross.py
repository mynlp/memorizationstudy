import pandas as pd
from pythia.utils.mmap_dataset import MMapIndexedDataset
import torch
import random
from tqdm import tqdm
from transformers import AutoTokenizer
import argparse
import pdb
from transformers import GPTNeoXForCausalLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(42)
model_size = "70m"
model_name = f"EleutherAI/pythia-{model_size}-deduped-v0"
CHECKPOINT = 143000
tokenizer = AutoTokenizer.from_pretrained(
  model_name,
  revision=f"step{CHECKPOINT}",
  cache_dir=f"./pythia-160m-deduped/step{CHECKPOINT}",
)
prefix = 'deduped_merge/document.bin'
context_size = 32
continuation_size = 16
results = pd.read_csv(f"generate_results/memorization_evals_{model_size}-deduped-v0_{context_size}_{context_size+continuation_size}_143000.csv", index_col=0)
model = GPTNeoXForCausalLM.from_pretrained(
        model_name,
        use_cache=False,
        revision=f'step143000',
    ).eval()
batch_size = 100
model = model.to_bettertransformer()
model = model.to(device)
model.generation_config.pad_token_id = model.generation_config.eos_token_id
model.generation_config.output_hidden_states = True
model.generation_config.output_attentions = True
model.generation_config.output_scores = True
model.generation_config.return_dict_in_generate = True
num_samples = 2000
memorized_results = {}
for i in range(continuation_size+1):
    memorized_results[str(i)] = results[results['score'] == i/continuation_size]
mmap_ds = MMapIndexedDataset(prefix, skip_warmup=True)

datasets = {}
for i in tqdm(range(10, continuation_size+1)):
    if len(memorized_results[str(i)]["idx"].tolist()) > num_samples:
        idx = random.sample(memorized_results[str(i)]["idx"].tolist(), num_samples)
    else:
        idx = random.sample(memorized_results[str(i)]["idx"].tolist())
    context_tokens = []
    for j in tqdm(idx):
        data = mmap_ds[j]
        context_tokens.append(data[:context_size+continuation_size].tolist())
    start = 0
    embedding_list = []
    context_tokens = torch.tensor(context_tokens).to(device)
    memorized_idx = []
    entropy = []
    for batch_idx in tqdm(range(0, num_samples, batch_size)):
        end = min(start+batch_size, num_samples)
        model_outputs = model.generate(context_tokens[start:end, :context_size], temperature=0.0, top_k=0, top_p=0,
                       max_length=context_size + continuation_size,
                       min_length=context_size + continuation_size)
        embeddings =  model_outputs.hidden_states[-1][-1]
        generated_sequence = model_outputs.sequences
        embedding_list.append(embeddings.cpu())
        logits = model_outputs["scores"]
        batched_entropy_at_idx = []
        for entropy_idx in range(continuation_size):
            probability_scores = torch.nn.functional.softmax(logits[entropy_idx], dim=1)
            entropy_scores = torch.distributions.Categorical(probs=probability_scores).entropy()
            batched_entropy_at_idx.append(entropy_scores)
        batched_entropy_at_idx = torch.stack(batched_entropy_at_idx, dim=1)
        continuation_alignment = context_tokens[start:end, context_size:] == generated_sequence[:, context_size:]
        continuation_alignment = continuation_alignment.float()
        memorized_idx.append(continuation_alignment.cpu())
        entropy.append(batched_entropy_at_idx.cpu())
        pdb.set_trace()
        start = end
    embeddings = torch.cat(embedding_list, dim=0)
    memorized_idx = torch.cat(memorized_idx, dim=0)
    entropy = torch.cat(entropy, dim=0)
    datasets[str(i)] = context_tokens
    torch.save(datasets[str(i)], f"cross_remembered/context_tokens_{continuation_size}_{i}_{model_size}.pt")
    torch.save(embeddings, f"cross_remembered/embeddings_{continuation_size}_{i}_{model_size}.pt")
    torch.save(memorized_idx, f"cross_remembered/memorized_idx_{continuation_size}_{i}_{model_size}.pt")
#
# paser = argparse.ArgumentParser()
# paser.add_argument("--distribution_idx", type=int, default=0)
# args = paser.parse_args()
#
# idx_70 = set(memorized_results_70["idx"].tolist())
#
# mmap_ds = MMapIndexedDataset(prefix, skip_warmup=True)
#
# # context_tokens = []
# # for i in tqdm(list(cross_all)):
# #     data = mmap_ds[i]
# #     context_tokens.extend(data.tolist())
# #     i += len(context_tokens)
# # context_tokens = torch.tensor(context_tokens)
# # torch.save(context_tokens, "cross_remembered/context_tokens.pt")
#
# context_tokens = []
# idx = 0
# # 举例，这是你不想包括的idx
# unmemorized = random.sample(unmemorized_idx, len(cross_all)*10)
# part_size = len(unmemorized) // 5
#
# # 创建3个子列表
# part1 = unmemorized[0:part_size]
# part2 = unmemorized[part_size:1*part_size]
# part3 = unmemorized[2*part_size:3*part_size]
# part4 = unmemorized[3*part_size:4*part_size]
# part5 = unmemorized[4*part_size:]
#
#
# # 第2步: 创建可用于抽样的索引列表
# # if args.distribution_idx == 0:
# #     excluded_idx = part1
# # elif args.distribution_idx == 1:
# #     excluded_idx = part2
# # elif args.distribution_idx == 2:
# #     excluded_idx = part3
# # elif args.distribution_idx == 3:
# #     excluded_idx = part4
# # elif args.distribution_idx == 4:
# #     excluded_idx = part5
# #
# # available_idx = [i for i in unmemorized_idx if i not in set(excluded_idx)]
# for i in tqdm(part1):
#     data = mmap_ds[i]
#     idx += 1
#     text = tokenizer.decode(data)
#     context_tokens.append([idx, text])
# df = pd.DataFrame(context_tokens, columns=["idx", "text"])
# df.to_json(f"cross_remembered/unmemorized_text_{args.distribution_idx}.json", index=False, orient='records', lines=True)




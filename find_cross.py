import pandas as pd
from pythia.utils.mmap_dataset import MMapIndexedDataset
import torch
import random
from tqdm import tqdm
from transformers import AutoTokenizer
import argparse
import pdb
from transformers import GPTNeoXForCausalLM, AutoTokenizer
import argparse

args = argparse.ArgumentParser()
args.add_argument("--model_size", type=str, default="6.9b")
args.add_argument("--context_size", type=int, default=128)
args.add_argument("--continuation_size", type=int, default=16)
args.add_argument("--seed", type=int, default=42)
args.add_argument("--epoch", type=int, default=40)
args.add_argument("--load_cache", type=bool, default=True)
args.add_argument("--batch_size", type=int, default=128)
args.add_argument("--num_samples", type=int, default=2000)
args = args.parse_args()
random.seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(42)
model_name = f"EleutherAI/pythia-{args.model_size}-deduped-v0"
prefix = 'deduped_merge/document.bin'
results = pd.read_csv(f"generate_results/memorization_evals_{args.model_size}-deduped-v0_{args.context_size}_{args.context_size+args.continuation_size}_143000.csv", index_col=0)
model = GPTNeoXForCausalLM.from_pretrained(
        model_name,
        use_cache=False,
        revision=f'step143000',
    ).eval()#model = model.to_bettertransformer()
#model = model.to(device)
if torch.cuda.is_available():
    device_ids = list(range(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model, device_ids=device_ids)
model.module.generation_config.pad_token_id = model.module.generation_config.eos_token_id
model.module.generation_config.output_hidden_states = True
model.module.generation_config.output_attentions = True
model.module.generation_config.output_scores = True
model.module.generation_config.return_dict_in_generate = True
memorized_results = {}
for i in range(args.continuation_size+1):
    memorized_results[str(i)] = results[results['score'] == i/args.continuation_size]
mmap_ds = MMapIndexedDataset(prefix, skip_warmup=True)

datasets = {}
for i in tqdm(range(args.continuation_size+1)):
    if len(memorized_results[str(i)]["idx"].tolist()) > args.num_samples:
        idx = random.sample(memorized_results[str(i)]["idx"].tolist(), args.num_samples)
    else:
        idx = random.sample(memorized_results[str(i)]["idx"].tolist())
    context_tokens = []
    for j in tqdm(idx):
        data = mmap_ds[j]
        context_tokens.append(data[:args.context_size+args.continuation_size].tolist())
    start = 0
    embedding_list = []
    context_tokens = torch.tensor(context_tokens).to(device)
    memorized_idx = []
    entropy = []
    for batch_idx in tqdm(range(0, args.num_samples, args.batch_size)):
        end = min(start+args.batch_size, args.num_samples)
        model_outputs = model.module.generate(context_tokens[start:end, :args.context_size], temperature=0.0, top_k=0, top_p=0,
                       max_length=args.context_size + args.continuation_size,
                       min_length=args.context_size + args.continuation_size)
        embeddings = model_outputs.hidden_states[-1][-1]
        generated_sequence = model_outputs.sequences
        embedding_list.append(embeddings.cpu())
        logits = model_outputs["scores"]
        batched_entropy_at_idx = []
        for entropy_idx in range(args.continuation_size):
            probability_scores = torch.nn.functional.softmax(logits[entropy_idx], dim=1)
            entropy_scores = torch.distributions.Categorical(probs=probability_scores).entropy()
            batched_entropy_at_idx.append(entropy_scores)
        batched_entropy_at_idx = torch.stack(batched_entropy_at_idx, dim=1)
        continuation_alignment = context_tokens[start:end, args.context_size:] == generated_sequence[:, args.context_size:]
        continuation_alignment = continuation_alignment.float()
        memorized_idx.append(continuation_alignment.cpu())
        entropy.append(batched_entropy_at_idx.cpu())
        start = end
    embeddings = torch.cat(embedding_list, dim=0)
    memorized_idx = torch.cat(memorized_idx, dim=0)
    entropy = torch.cat(entropy, dim=0)
    datasets[str(i)] = context_tokens
    torch.save(datasets[str(i)], f"cross_remembered/context_tokens_{args.context_size}_{args.continuation_size}_{i}_{args.model_size}.pt")
    torch.save(embeddings, f"cross_remembered/embeddings_{args.context_size}_{args.continuation_size}_{i}_{args.model_size}.pt")
    torch.save(memorized_idx, f"cross_remembered/memorized_idx_{args.context_size}_{args.continuation_size}_{i}_{args.model_size}.pt")
    torch.save(entropy, f"cross_remembered/entropy_{args.context_size}_{args.continuation_size}_{i}_{args.model_size}.pt")




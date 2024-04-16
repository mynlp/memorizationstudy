import torch
from transformers import AutoTokenizer, pipeline, StoppingCriteria, StoppingCriteriaList
import argparse
import pandas as pd
from pythia.utils.mmap_dataset import MMapIndexedDataset
import random

random.seed(42)
args = argparse.ArgumentParser()
args.add_argument("--model_size", type=str, default="1.4b")
args.add_argument("--max_new_tokens", type=int, default=2048)
args.add_argument("--stop_token", type=str, default="<|stop|>")
args.add_argument("--context", type=int, default=32)
args.add_argument("--continuation", type=int, default=16)
args = args.parse_args()

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

model_name = f"lambdalabs/pythia-{args.model_size}-deduped-synthetic-instruct"
max_new_tokens = 2048
df = pd.read_csv(f"generate_results/memorization_evals_1b-deduped-v0_{args.context}_{args.context+args.continuation}_143000.csv", index_col=0)

prefix = 'deduped_merge/document.bin'
print(prefix)
buff_size = 2049*1024*2
print("Building dataset")
mmap_ds = MMapIndexedDataset(prefix, skip_warmup=True)
original_tokenizer = AutoTokenizer.from_pretrained(
  f"EleutherAI/pythia-1b-deduped-v0",
  revision=f"step143000",
  cache_dir=f"./pythia-1b-deduped/step143000",
)
class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_ids: list):
        self.keywords = keywords_ids

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        if input_ids[0][-1] in self.keywords:
            return True
        return False


tokenizer = AutoTokenizer.from_pretrained(
    model_name,
)

memorized_dict = df[df['score'] == 1]
unmemorized_dict = df[df['score'] == 0]
idx_full_memorization = memorized_dict["idx"].tolist()
idx_not_full_memorization = unmemorized_dict["idx"].tolist()
memorized_idx = random.sample(idx_full_memorization, 1000)
unmemorized_idx = random.sample(idx_not_full_memorization, 1000)
memorized_batched_context_tokens = []
unmemorized_batched_context_tokens = []
for idx in memorized_idx:
    data = mmap_ds[idx]
    context_tokens = original_tokenizer.decode(data[:args.context+args.continuation].tolist())
    memorized_batched_context_tokens.append(context_tokens)
for idx in unmemorized_idx:
    data = mmap_ds[idx]
    context_tokens = original_tokenizer.decode(data[:args.context+args.continuation].tolist())
    unmemorized_batched_context_tokens.append(context_tokens)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_tokens([args.stop_token])
stop_ids = [tokenizer.encode(w)[0] for w in [args.stop_token]]
stop_criteria = KeywordsStoppingCriteria(stop_ids)

generator = pipeline(
    "text-generation",
    model=model_name,
    device=device,
    max_new_tokens=args.max_new_tokens,
    torch_dtype=torch.float16,
    stopping_criteria=StoppingCriteriaList([stop_criteria]),
)

example = f"Do you remember the following sentence?\n{memorized_batched_context_tokens[0]}. \nReply with only Yes or No"
text = "Question: {}\nAnswer:".format(example)
result = generator(
    text,
    num_return_sequences=1,
)
output = result[0]["generated_text"]
print(output)

example = f"Do you remember the following sentence?\n{unmemorized_batched_context_tokens[0]}. \nReply with only Yes or No"
text = "Question: {}\nAnswer:".format(example)
result = generator(
    text,
    num_return_sequences=1,
)
output = result[0]["generated_text"]
print(output)

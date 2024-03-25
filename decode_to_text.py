import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling


def batchfy(data):
  input_batch = []
  for input_ids in (data):
      input_batch.append(input_ids)
  return {"input_ids": input_batch}

data = torch.load("cross_remembered/context_tokens.pt")
model_name = "EleutherAI/pythia-160m-deduped-v0"
CHECKPOINT = 143000
tokenizer = AutoTokenizer.from_pretrained(
  model_name,
  revision=f"step{CHECKPOINT}",
  cache_dir=f"./pythia-160m-deduped/step{CHECKPOINT}",
)

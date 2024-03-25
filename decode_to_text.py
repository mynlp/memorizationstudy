import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling


data = torch.load("cross_remembered/context_tokens.pt")
model_name = "EleutherAI/pythia-160m-deduped-v0"
CHECKPOINT = 143000
tokenizer = AutoTokenizer.from_pretrained(
  model_name,
  revision=f"step{CHECKPOINT}",
  cache_dir=f"./pythia-160m-deduped/step{CHECKPOINT}",
)

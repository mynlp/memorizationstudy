import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm


def batchfy(data):
  input_batch = []
  for input_ids in (data):
      input_batch.append(input_ids)
  return {"input_ids": input_batch}

data = torch.load("cross_remembered/context_tokens.pt")
data = data.view(-1, 2049)
model_name = "EleutherAI/pythia-160m-deduped-v0"
CHECKPOINT = 143000
tokenizer = AutoTokenizer.from_pretrained(
  model_name,
  revision=f"step{CHECKPOINT}",
  cache_dir=f"./pythia-160m-deduped/step{CHECKPOINT}",
)
result = []
for idx, line in tqdm(enumerate(data)):
  text = tokenizer.decode(line)
  result.append([idx, text])
df = pd.DataFrame(result, columns=["idx", "text"])
df.to_json("cross_remembered/memorized_text.json", index=False, orient='records', lines=True))

column_a_as_text = '\n'.join(df["text"].astype(str))
f = open("cross_remembered/memorized_text.txt", "w")
f.write(column_a_as_text)
f.close()



dataset = datasets.load_dataset("json", data_files="cross_remembered/memorized_text.json")

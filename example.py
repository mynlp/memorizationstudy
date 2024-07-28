from datasets import load_dataset
import pandas as pd
import torch
import numpy as np
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import pandas as pd

REPO_ID = "Parallaxixs/ARRJuneData"
FILENAME = "data_sample.csv"
print("Data Download")
df = pd.read_csv(
    hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset")
)

numpy_data = df.to_numpy(dtype=int)
data_tensor = torch.from_numpy(numpy_data).int()

model_size = "410m"#model size parameter
batch_size = 10#generetion batch size
context = 32#number of context tokens
continuation = 16#number of continuation tokens
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GPTNeoXForCausalLM.from_pretrained(
    f"EleutherAI/pythia-{model_size}",
    revision=f'step143000',
)

if torch.cuda.is_available():
    model = model.half()

model = model.eval().to(device)

model = model.to_bettertransformer()

tokenizer = AutoTokenizer.from_pretrained(
  f"EleutherAI/pythia-{model_size}-deduped",
  revision="step143000",
  cache_dir=f"./pythia-{model_size}-deduped/step143000",
)

num_batches = len(data_tensor) // batch_size
# Take care of the last batch if it doesn't align with the `batch_size`
if len(data_tensor) % batch_size != 0:
    num_batches += 1
accuracy_list = []
for i in tqdm(range(num_batches)):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, len(data_tensor))
    batch_data = data_tensor[start_idx:end_idx]
    context_tokens = torch.stack([sample[:context] for sample in batch_data]).cuda()
    true_continuation = torch.stack([sample[context:context + continuation] for sample in batch_data]).cuda()
    with torch.no_grad():
        generations = model.generate(context_tokens, temperature=0.0, top_k=0, top_p=0,
                                     max_length=context + continuation,
                                     min_length=context + continuation)
        accuracies = (true_continuation == generations[:, context:context + continuation]).float().sum(
            dim=1).tolist()##compare to cacualte the memorization scores
        accuracy_list.extend(accuracies)
accuracy_list = torch.tensor(accuracy_list) #the memorization score for each sample




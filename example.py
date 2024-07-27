from datasets import load_dataset
import pandas as pd
import torch
import numpy as np
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from tqdm import tqdm

dataset = load_dataset("Parallaxixs/ARRJuneData")
df = pd.DataFrame(dataset['train'])

numpy_data = df.to_numpy(dtype=np.int)
data_tensor = torch.from_numpy(numpy_data).int()

model_size = "410m"
batch_size = 10
context = 32
continuation = 16

model = GPTNeoXForCausalLM.from_pretrained(
    f"EleutherAI/pythia-{model_size}",
    revision=f'step143000',
).half().eval().cuda(0)
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
            dim=1).tolist()
        accuracy_list.extend(accuracies)
accuracy_list = torch.tensor(accuracy_list)




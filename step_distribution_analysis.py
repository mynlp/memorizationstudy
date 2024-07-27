from pythia.utils.mmap_dataset import MMapIndexedDataset
from utils import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
import pandas as pd
import json
from tqdm import tqdm


random.seed(42)
small_model_size = "70m"
large_model_size = "410m"
context = 32
continuation = 16
prefix = 'deduped_merge/document.bin'
print(prefix)
buff_size = 2049*1024*2
print("Building dataset")
mmap_ds = MMapIndexedDataset(prefix, skip_warmup=True)


df_small = pd.read_csv(f"generate_results/memorization_evals_{small_model_size}-deduped-v0_{context}_{context+continuation}_143000.csv", index_col=0)
df_small_memorized = df_small[df_small["score"] == 0.5]
small_memorized_idx = df_small_memorized.index

model = GPTNeoXForCausalLM.from_pretrained(
    f"EleutherAI/pythia-{small_model_size}",
    revision=f'step143000',
).eval().cuda(0)
model = model.to_bettertransformer()

tokenizer = AutoTokenizer.from_pretrained(
  f"EleutherAI/pythia-{small_model_size}-deduped",
  revision="step143000",
  cache_dir=f"./pythia-{small_model_size}-deduped/step143000",
)
accuracy = []
for idx in tqdm(small_memorized_idx[:100]):
    data = mmap_ds[idx]
    context_tokens = data[:context].tolist()
    true_continuation = data[context:context + continuation].tolist()
    context_tokens = torch.tensor(context_tokens).unsqueeze(0).cuda()
    true_continuation = torch.tensor(true_continuation).unsqueeze(0).cuda()
    with torch.no_grad():
        if isinstance(model, torch.nn.DataParallel):
            generations = model.module.generate(context_tokens, temperature=0.0, top_k=0, top_p=0,
                                                max_length=context + continuation,
                                                min_length=context + continuation)
        else:
            generations = model.generate(context_tokens, temperature=0.0, top_k=0, top_p=0,
                                         max_length=context + continuation,
                                         min_length=context + continuation)
    accuracies = (true_continuation == generations[:, context:context + continuation]).float()
    accuracy.append(accuracies.tolist())
    print(context_tokens)
    print(true_continuation)
    print(continuation)
    print("=====================================================")
accuracy = torch.tensor(accuracy)
sum_across_16 = accuracy.sum(dim=-1)
print(sum_across_16)

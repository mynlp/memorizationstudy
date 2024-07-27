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
batch_size = 32
context_tokens_list = []
true_continuation_list = []

for idx in tqdm(small_memorized_idx[:1000]):
    data = mmap_ds[idx]

    context_tokens_list.append(data[:context].tolist())
    true_continuation_list.append(data[context:context + continuation].tolist())

    # Check if batch size is reached
    if len(context_tokens_list) >= batch_size:
        # Process batch here and do what you want with it
        context_tokens_batch = torch.tensor(context_tokens_list).cuda()
        true_continuation_batch = torch.tensor(true_continuation_list).cuda()
        with torch.no_grad():
            if isinstance(model, torch.nn.DataParallel):
                generations = model.module.generate(context_tokens_batch, temperature=0.0, top_k=0, top_p=0,
                                                    max_length=context + continuation,
                                                    min_length=context + continuation)
            else:
                generations = model.generate(context_tokens_batch, temperature=0.0, top_k=0, top_p=0,
                                             max_length=context + continuation,
                                             min_length=context + continuation)
        accuracies = (true_continuation_batch == generations[:, context:context + continuation]).float()
        accuracy.append(accuracies.tolist())
        print(context_tokens_batch)
        print(true_continuation_batch)
        print(continuation)
        print("=====================================================")
        context_tokens_list = []
        true_continuation_list = []
accuracy = torch.tensor(accuracy)


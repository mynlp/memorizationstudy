import torch
from pythia.utils.mmap_dataset import MMapIndexedDataset
from utils import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
import pandas as pd
from tqdm import tqdm


random.seed(42)
small_model_size = "410m"
context = 32
continuation = 16
prefix = 'deduped_merge/document.bin'
print(prefix)
buff_size = 2049*1024*2
print("Building dataset")
mmap_ds = MMapIndexedDataset(prefix, skip_warmup=True)
batch_size = 10
df_small = pd.read_csv(f"generate_results/memorization_evals_{small_model_size}-deduped-v0_{context}_{context+continuation}_143000.csv", index_col=0)

scores = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
data_list = []
for score in scores:
    df_small_memorized = df_small[df_small["score"] == score]
    small_memorized_idx = df_small_memorized.index
    for idx in tqdm(small_memorized_idx[:500]):
        data = mmap_ds[idx]
        data_list.append(data.tolist())
data = torch.tensor(data_list)
data = data.cpu().numpy() if data.is_cuda else data.numpy()
df = pd.DataFrame(data)
df.to_csv('data_sample.csv', index=False)
df = pd.read_csv('data_sample.csv')
data_numpy = df.values
data_tensor = torch.from_numpy(data_numpy)
data_tensor = data_tensor.int()

model = GPTNeoXForCausalLM.from_pretrained(
    f"EleutherAI/pythia-{small_model_size}",
    revision=f'step143000',
).half().eval().cuda(0)
model = model.to_bettertransformer()

tokenizer = AutoTokenizer.from_pretrained(
  f"EleutherAI/pythia-{small_model_size}-deduped",
  revision="step143000",
  cache_dir=f"./pythia-{small_model_size}-deduped/step143000",
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
    context_tokens = torch.tensor([sample[:context] for sample in batch_data]).cuda()
    true_continuation = torch.tensor([sample[context:context + continuation] for sample in batch_data]).cuda()
    with torch.no_grad():
        generations = model.generate(context_tokens, temperature=0.0, top_k=0, top_p=0,
                                     max_length=context + continuation,
                                     min_length=context + continuation)
        accuracies = (true_continuation == generations[:, context:context + continuation]).float().sum(
            dim=1).tolist()
        accuracy_list.extend(accuracies)
accuracy_list = torch.tensor(accuracy_list)
for idx, score in enumerate(scores):
    temp = accuracy_list/continuation == score
    print(f"Number of Samples equal to score {score}: {temp.sum()}")



# accuracy = []
# batch_size = 32
# context_tokens_list = []
# true_continuation_list = []
#
# for idx in tqdm(small_memorized_idx[:10000]):
#     data = mmap_ds[idx]
#
#     context_tokens_list.append(data[:context].tolist())
#     true_continuation_list.append(data[context:context + continuation].tolist())
#
#     # Check if batch size is reached
#     if len(context_tokens_list) >= batch_size:
#         # Process batch here and do what you want with it
#         context_tokens_batch = torch.tensor(context_tokens_list).cuda()
#         true_continuation_batch = torch.tensor(true_continuation_list).cuda()
#         with torch.no_grad():
#             if isinstance(model, torch.nn.DataParallel):
#                 generations = model.module.generate(context_tokens_batch, temperature=0.0, top_k=0, top_p=0,
#                                                     max_length=context + continuation,
#                                                     min_length=context + continuation)
#             else:
#                 generations = model.generate(context_tokens_batch, temperature=0.0, top_k=0, top_p=0,
#                                              max_length=context + continuation,
#                                              min_length=context + continuation)
#         accuracies = (true_continuation_batch == generations[:, context:context + continuation]).float()
#         accuracy.append(accuracies.tolist())
#         print(context_tokens_batch)
#         print(true_continuation_batch)
#         print(continuation)
#         print("=====================================================")
#         context_tokens_list = []
#         true_continuation_list = []
# accuracy = torch.tensor(accuracy)
# accuracy = torch.cat([ accuracy_batch for accuracy_batch in accuracy],dim=0)
# print(accuracy.sum(dim=0)/accuracy.shape[0])

import pandas as pd
from transformers import GPTNeoXForCausalLM, AutoTokenizer,  AutoModelForCausalLM,  LogitsProcessorList, MinLengthLogitsProcessor, StoppingCriteriaList,  MaxLengthCriteria
from pythia.utils.mmap_dataset import MMapIndexedDataset
from transformers import GPTNeoXForCausalLM, AutoTokenizer
import torch
from utils import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random

random.seed(42)
model_name = "EleutherAI/pythia-70m-deduped-v0"
CHECKPOINT= 143000
model = GPTNeoXForCausalLM.from_pretrained(
  model_name,
  revision=f"step{CHECKPOINT}",
  cache_dir=f"./pythia-70m-deduped/step{CHECKPOINT}",
)
model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(
  model_name,
  revision=f"step{CHECKPOINT}",
  cache_dir=f"./pythia-70m-deduped/step{CHECKPOINT}",
)
model.generation_config.pad_token_id = model.generation_config.eos_token_id
model.generation_config.output_hidden_states = True
model.generation_config.output_attentions = True
model.generation_config.output_scores = True
model.generation_config.return_dict_in_generate = True


prefix = 'undeduped_merge/document.bin'
if "deduped" in model_name:
    prefix = 'deduped_merge/document.bin'
print(prefix)
buff_size = 2049*1024*2
print("Building dataset")
mmap_ds = MMapIndexedDataset(prefix, skip_warmup=True)

df = pd.read_csv("generate_results/memorization_evals_70m-deduped-v0_32_48_143000.csv", names=["idx", "score"])
df_full_memorization = df[df['score'] == 1]
df_not_full_memorization = df[df['score'] == 0]

idx_full_memorization = df_full_memorization["idx"].tolist()
idx_not_full_memorization = df_not_full_memorization["idx"].tolist()

stragety = "mean_hidden_state"
for num_points in [100, 200, 300 ,400, 500,]:
  generations_full_memo, accuracies_full_memo = embedding_obtain(mmap_ds, model,  random.sample(idx_full_memorization,num_points), 32, 16)
  generations_not_full, accuracies_not_full = embedding_obtain(mmap_ds, model,  random.sample(idx_not_full_memorization,num_points), 32, 16)

  # last hidden state
  if stragety == "last_hidden_state":
    embedding = generations_full_memo.hidden_states[-1][-1].squeeze().cpu().numpy()
    embedding_not_full = generations_not_full.hidden_states[-1][-1].squeeze().cpu().numpy()
  # mean pooling hidden state of all continuation token at last year
  elif stragety == "mean_hidden_state":
    embedding = torch.stack([x[-1] for x in generations_full_memo.hidden_states[1:]]).mean(0).squeeze().cpu().numpy()
    embedding_not_full = torch.stack([x[-1] for x in generations_not_full.hidden_states[1:]]).mean(0).squeeze().cpu().numpy()
  # max pooling hidden state of all continuation token at last year
  elif stragety == "max_hidden_state":
    embedding = torch.stack([x[-1] for x in generations_full_memo.hidden_states[1:]]).max(0)[0].values.squeeze().cpu().numpy()
    embedding_not_full = torch.stack([x[-1] for x in generations_not_full.hidden_states[1:]]).max(0)[0].values.squeeze().cpu().numpy()
  data = np.vstack((embedding, embedding_not_full))

  tsne = TSNE(n_components=2, random_state=42)
  data_tsne = tsne.fit_transform(data)

  plt.figure(figsize=(8, 6))

  plt.scatter(data_tsne[:num_points, 0], data_tsne[:num_points, 1], color='blue', label='A')
  plt.scatter(data_tsne[num_points:, 0], data_tsne[num_points:, 1], color='red', label='B')
  plt.title('t-SNE Visualization')
  plt.legend()
  plt.savefig(f'tsne_visualization_{num_points}_{stragety}.png')
  plt.show()








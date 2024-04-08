import pandas as pd
from transformers import GPTNeoXForCausalLM, AutoTokenizer,  AutoModelForCausalLM,  LogitsProcessorList, MinLengthLogitsProcessor, StoppingCriteriaList,  MaxLengthCriteria
from pythia.utils.mmap_dataset import MMapIndexedDataset
from transformers import GPTNeoXForCausalLM, AutoTokenizer
import torch
from utils import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA
import seaborn as sns
import torch.nn.functional as F
import matplotlib.cm as cm


random.seed(42)
model_name = "EleutherAI/pythia-70m-deduped-v0"
CHECKPOINT= 143000
context = 32
continuation = 16
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
colors = cm.rainbow(np.linspace(0, 1, continuation + 1))

df = pd.read_csv(f"generate_results/memorization_evals_70m-deduped-v0_{context}_{context+continuation}_143000.csv", index_col=0)
# memorized_dict = {"1": df[df['score'] == 1], "0.9":df[df['score'].between(int(continuation*0.88)/continuation, int(continuation*0.92)/continuation)],
#                   "0.8":df[df['score'].between(int(continuation*0.78)/continuation, int(continuation*0.82)/continuation)],"0.7":df[df['score'].between(int(continuation*0.68)/continuation, int(continuation*0.72)/continuation)],
#                   "0.6":df[df['score'].between(int(continuation*0.58)/continuation, int(continuation*0.62)/continuation)], "0.5":df[df['score'].between(int(continuation*0.48)/continuation, int(continuation*0.52)/continuation)],
#                   "0.4":df[df['score'].between(int(continuation*0.38)/continuation, int(continuation*0.42)/continuation)], "0.3":df[df['score'].between(int(continuation*0.28)/continuation, int(continuation*0.32)/continuation)],
#                   "0.2":df[df['score'].between(int(continuation*0.18)/continuation, int(continuation*0.22)/continuation)], "0.1":df[df['score'].between(int(continuation*0.08)/continuation, int(continuation*0.12)/continuation)],
#                   "0":df[df['score'] == 0]}

memorized_dict = {}
for key in range(continuation+1):
    memorized_dict[str(key)] = df[df['score'] == key/continuation]
idx = []
for key in memorized_dict.keys():
    idx.append(memorized_dict[key]["idx"].tolist())

stragety = "dynamics"
num_points = 100
generations = []
accuracies = []
for memorized_idx in idx:
    generation, accuracie = embedding_obtain(mmap_ds, model,  random.sample(memorized_idx,num_points), 32, continuation)
    generations.append(generation)
    accuracies.append(accuracie)

# last hidden state
context_embeddings = []
for generation in generations:
    context_embeddings.append(generation.hidden_states[0][-1])

distance_list = {}
for i in range(continuation+1):
    distance_list[i] = []
for token in range(2, continuation+1):
    predicted_embeddings = []
    for generation in generations:
        predicted_embeddings.append(torch.stack([x[-1] for x in generation.hidden_states[1:token]]).squeeze().transpose(0, 1) if token != 2 else torch.stack([x[-1] for x in generation.hidden_states[1:token]]).squeeze().unsqueeze(dim=1))
    averaged_embedding = []
    for context_embedding, predicted_embedding in zip(context_embeddings, predicted_embeddings):
        averaged_embedding.append(torch.concat((context_embedding, predicted_embedding), dim=1).mean(0).mean(0))
    for i in range(continuation+1):
        distance = torch.dist(averaged_embedding[int(continuation/2)], averaged_embedding[i])
        distance_list[i].append(float(distance))
    embeddings = []
    for embedding in predicted_embeddings:
        embeddings.append(embedding.mean(0)[token-2])

    names = [f"{i}" for i in range(continuation+1)]

    # 创建一个空的 DataFrame 用于保存结果
    plt.figure(figsize=(8, 6))
    similarities = pd.DataFrame(index=names, columns=names)
    for i in range(len(embeddings)):
        for j in range(i, len(embeddings)):
            similarity = F.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0)).item()
            similarities.loc[names[i], names[j]] = similarity
            similarities.loc[names[j], names[i]] = similarity
    plt.figure(figsize=(10, 10))
    sns.heatmap(similarities.astype(float), annot=True, fmt=".2f", annot_kws={"size": 10}, square=True, cmap='hot',  xticklabels=True, yticklabels=True)
    plt.title(f'Embedding Similarities_{token}')
    plt.savefig(f'embedding_figure/embedding_similarities_{token}.png')
    plt.show()

    plt.figure(figsize=(8, 6))
    distances = pd.DataFrame(index=names, columns=names)
    for i in range(len(embeddings)):
        for j in range(i, len(embeddings)):
            distance = torch.dist(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0), p=2).item()
            distances.loc[names[i], names[j]] = distance
            distances.loc[names[j], names[i]] = distance
    plt.figure(figsize=(10, 10))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(distances.astype(float), annot=True, fmt=".2f",  annot_kws={"size": 10}, square=True, cmap=cmap,  xticklabels=True, yticklabels=True)
    plt.title(f'Embedding Distances_{token}')
    plt.savefig(f'embedding_figure/embedding_distances_{token}.png')
    plt.show()
    plt.figure(figsize=(8, 6))
    all_embeddings = np.stack([embedding.cpu().numpy() for embedding in embeddings], axis=0)
    n_samples = all_embeddings.shape[0]
    pca = PCA(n_components=2, random_state=42)
    tsne_embeddings = pca.fit_transform(all_embeddings)
    plt.xlim(-60, 60)
    plt.ylim(-60, 60)
    for i in range(continuation+1):
        plt.scatter(tsne_embeddings[i, 0], tsne_embeddings[i, 1], color=colors[i], label=f'{i}')
    for i in range(continuation+1):
        print(f"Distance between half and {i}: {distance_list[i]}")
    plt.title('PCA Visualization')
    plt.legend()
    plt.savefig(f'embedding_figure/embedding_drift_step_{token}.png')
    plt.show()
print("distance across steps:")
for i in range(continuation+1):
    print(f"Distance between half and {i}: {distance_list[i]}")









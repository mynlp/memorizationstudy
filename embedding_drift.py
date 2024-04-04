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

random.seed(42)
model_name = "EleutherAI/pythia-70m-deduped-v0"
CHECKPOINT= 143000
continuation_size = 16
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

df = pd.read_csv("generate_results/memorization_evals_70m-deduped-v0_32_48_143000.csv", index_col=0)
memorized_dict = {"1": df[df['score'] == 1], "0.9":df[df['score'] == int(continuation_size*0.9)/continuation_size], "0.8":df[df['score'] == int(continuation_size*0.8)/continuation_size],
                  "0.7":df[df['score'] == int(continuation_size*0.7)/continuation_size], "0.6":df[df['score'] == int(continuation_size*0.6)/continuation_size], "0.5":df[df['score'] == int(continuation_size*0.5)/continuation_size],
                  "0.4":df[df['score'] == int(continuation_size*0.4)/continuation_size], "0.3":df[df['score'] == int(continuation_size*0.3)/continuation_size], "0.2":df[df['score'] == int(continuation_size*0.2)/continuation_size],
                  "0.1":df[df['score'] == int(continuation_size*0.1)/continuation_size], "0":df[df['score'] == 0]}

idx_full_memorization = memorized_dict["1"]["idx"].tolist()
idx_ninety_memorization = memorized_dict["0.9"]["idx"].tolist()
idx_eighty_memorization = memorized_dict["0.8"]["idx"].tolist()
idx_seventy_memorization = memorized_dict["0.7"]["idx"].tolist()
idx_sixty_memorization = memorized_dict["0.6"]["idx"].tolist()
idx_half_memorization = memorized_dict["0.5"]["idx"].tolist()
idx_fourty_memorization = memorized_dict["0.4"]["idx"].tolist()
idx_thirty_memorization = memorized_dict["0.3"]["idx"].tolist()
idx_twenty_memorization = memorized_dict["0.2"]["idx"].tolist()
idx_ten_memorization = memorized_dict["0.1"]["idx"].tolist()
idx_not_full_memorization = memorized_dict["0"]["idx"].tolist()

stragety = "dynamics"
num_points = 100
generations_full_memo, accuracies_full_memo = embedding_obtain(mmap_ds, model,  random.sample(idx_full_memorization,num_points), 32, 16)
generations_ninety_memo, accuracies_ninety_memo = embedding_obtain(mmap_ds, model,  random.sample(idx_ninety_memorization,num_points), 32, 16)
generations_eighty_memo, accuracies_eighty_memo = embedding_obtain(mmap_ds, model,  random.sample(idx_eighty_memorization,num_points), 32, 16)
generations_seventy_full, accuracies_seventy_full = embedding_obtain(mmap_ds, model,  random.sample(idx_seventy_memorization,num_points), 32, 16)
generations_sixty_memo, accuracies_sixty_memo = embedding_obtain(mmap_ds, model,  random.sample(idx_sixty_memorization,num_points), 32, 16)
generations_half_memo, accuracies_half_memo = embedding_obtain(mmap_ds, model,  random.sample(idx_half_memorization,num_points), 32, 16)
generations_fourty_full, accuracies_fourty_full = embedding_obtain(mmap_ds, model,  random.sample(idx_fourty_memorization,num_points), 32, 16)
generations_thirty_memo, accuracies_thirty_memo = embedding_obtain(mmap_ds, model,  random.sample(idx_thirty_memorization,num_points), 32, 16)
generations_twenty_memo, accuracies_twenty_memo = embedding_obtain(mmap_ds, model,  random.sample(idx_twenty_memorization,num_points), 32, 16)
generations_ten_memo, accuracies_ten_memo = embedding_obtain(mmap_ds, model,  random.sample(idx_ten_memorization,num_points), 32, 16)
generations_zero_full, accuracies_zero_full = embedding_obtain(mmap_ds, model,  random.sample(idx_not_full_memorization,num_points), 32, 16)

# last hidden state
context_embedding_full = generations_full_memo.hidden_states[0][-1]
context_embedding_ninety = generations_ninety_memo.hidden_states[0][-1]
context_embedding_eighty = generations_eighty_memo.hidden_states[0][-1]
context_embedding_seventy = generations_seventy_full.hidden_states[0][-1]
context_embedding_sixty = generations_sixty_memo.hidden_states[0][-1]
context_embedding_half = generations_half_memo.hidden_states[0][-1]
context_embedding_fourty = generations_fourty_full.hidden_states[0][-1]
context_embedding_thirty = generations_thirty_memo.hidden_states[0][-1]
context_embedding_twenty = generations_twenty_memo.hidden_states[0][-1]
context_embedding_ten = generations_ten_memo.hidden_states[0][-1]
context_embedding_zero = generations_zero_full.hidden_states[0][-1]

distance_list_full = []
distance_list_ninety = []
distance_list_eighty = []
distance_list_seventy = []
distance_list_sixty = []
distance_list_half = []
distance_list_fourty = []
distance_list_thirty = []
distance_list_twenty = []
distance_list_ten = []
distance_list_zero = []

for token in range(2, 17):
    plt.figure(figsize=(8, 6))
    predicted_embedding_full = torch.stack([x[-1] for x in generations_full_memo.hidden_states[1:token]]).squeeze().transpose(0, 1) if token != 2 else torch.stack([x[-1] for x in generations_full_memo.hidden_states[1:token]]).squeeze().unsqueeze(dim=1)
    predicted_embedding_ninety = torch.stack([x[-1] for x in generations_ninety_memo.hidden_states[1:token]]).squeeze().transpose(0, 1) if token != 2 else torch.stack([x[-1] for x in generations_ninety_memo.hidden_states[1:token]]).squeeze().unsqueeze(dim=1)
    predicted_embedding_eighty = torch.stack([x[-1] for x in generations_eighty_memo.hidden_states[1:token]]).squeeze().transpose(0, 1) if token != 2 else torch.stack([x[-1] for x in generations_eighty_memo.hidden_states[1:token]]).squeeze().unsqueeze(dim=1)
    predicted_embedding_seventy = torch.stack([x[-1] for x in generations_seventy_full.hidden_states[1:token]]).squeeze().transpose(0, 1) if token != 2 else torch.stack([x[-1] for x in generations_seventy_full.hidden_states[1:token]]).squeeze().unsqueeze(dim=1)
    predicted_embedding_sixty = torch.stack([x[-1] for x in generations_sixty_memo.hidden_states[1:token]]).squeeze().transpose(0, 1) if token != 2 else torch.stack([x[-1] for x in generations_sixty_memo.hidden_states[1:token]]).squeeze().unsqueeze(dim=1)
    predicted_embedding_half = torch.stack([x[-1] for x in generations_half_memo.hidden_states[1:token]]).squeeze().transpose(0, 1) if token != 2 else torch.stack([x[-1] for x in generations_half_memo.hidden_states[1:token]]).squeeze().unsqueeze(dim=1)
    predicted_embedding_forty = torch.stack([x[-1] for x in generations_fourty_full.hidden_states[1:token]]).squeeze().transpose(0, 1) if token != 2 else torch.stack([x[-1] for x in generations_fourty_full.hidden_states[1:token]]).squeeze().unsqueeze(dim=1)
    predicted_embedding_thirty = torch.stack([x[-1] for x in generations_thirty_memo.hidden_states[1:token]]).squeeze().transpose(0, 1) if token != 2 else torch.stack([x[-1] for x in generations_thirty_memo.hidden_states[1:token]]).squeeze().unsqueeze(dim=1)
    predicted_embedding_twenty = torch.stack([x[-1] for x in generations_twenty_memo.hidden_states[1:token]]).squeeze().transpose(0, 1) if token != 2 else torch.stack([x[-1] for x in generations_twenty_memo.hidden_states[1:token]]).squeeze().unsqueeze(dim=1)
    predicted_embedding_ten = torch.stack([x[-1] for x in generations_ten_memo.hidden_states[1:token]]).squeeze().transpose(0, 1) if token != 2 else torch.stack([x[-1] for x in generations_ten_memo.hidden_states[1:token]]).squeeze().unsqueeze(dim=1)
    predicted_embedding_zero =  torch.stack([x[-1] for x in generations_zero_full.hidden_states[1:token]]).squeeze().transpose(0, 1)  if token != 2 else torch.stack([x[-1] for x in generations_zero_full.hidden_states[1:token]]).squeeze().unsqueeze(dim=1)

    averaged_embedding_full = torch.concat((context_embedding_full, predicted_embedding_full), dim=1).mean(0).mean(0)
    averaged_embedding_ninety = torch.concat((context_embedding_ninety, predicted_embedding_ninety), dim=1).mean(0).mean(0)
    averaged_embedding_eighty = torch.concat((context_embedding_eighty, predicted_embedding_eighty), dim=1).mean(0).mean(0)
    averaged_embedding_seventy = torch.concat((context_embedding_seventy, predicted_embedding_seventy), dim=1).mean(0).mean(0)
    averaged_embedding_sixty = torch.concat((context_embedding_sixty, predicted_embedding_sixty), dim=1).mean(0).mean(0)
    averaged_embedding_half = torch.concat((context_embedding_half, predicted_embedding_half), dim=1).mean(0).mean(0)
    averaged_embedding_forty = torch.concat((context_embedding_fourty, predicted_embedding_forty), dim=1).mean(0).mean(0)
    averaged_embedding_thirty = torch.concat((context_embedding_thirty, predicted_embedding_thirty), dim=1).mean(0).mean(0)
    averaged_embedding_twenty = torch.concat((context_embedding_twenty, predicted_embedding_twenty), dim=1).mean(0).mean(0)
    averaged_embedding_ten = torch.concat((context_embedding_ten, predicted_embedding_ten), dim=1).mean(0).mean(0)
    averaged_embedding_zero = torch.concat((context_embedding_zero, predicted_embedding_zero), dim=1).mean(0).mean(0)
    distance_list_full.append(averaged_embedding_full)
    distance_list_ninety.append(averaged_embedding_ninety)
    distance_list_eighty.append(averaged_embedding_eighty)
    distance_list_seventy.append(averaged_embedding_seventy)
    distance_list_sixty.append(averaged_embedding_sixty)
    distance_list_half.append(averaged_embedding_half)
    distance_list_fourty.append(averaged_embedding_forty)
    distance_list_thirty.append(averaged_embedding_thirty)
    distance_list_twenty.append(averaged_embedding_twenty)
    distance_list_ten.append(averaged_embedding_ten)
    distance_list_zero.append(averaged_embedding_zero)
    distance_full = torch.dist(averaged_embedding_half, averaged_embedding_full)
    distance_ninety = torch.dist(averaged_embedding_half, averaged_embedding_ninety)
    distance_eighty = torch.dist(averaged_embedding_half, averaged_embedding_eighty)
    distance_seventy = torch.dist(averaged_embedding_half, averaged_embedding_seventy)
    distance_sixty = torch.dist(averaged_embedding_half, averaged_embedding_sixty)
    distance_fourty = torch.dist(averaged_embedding_half, averaged_embedding_forty)
    distance_thirty = torch.dist(averaged_embedding_half, averaged_embedding_thirty)
    distance_twenty = torch.dist(averaged_embedding_half, averaged_embedding_twenty)
    distance_ten = torch.dist(averaged_embedding_half, averaged_embedding_ten)
    distance_zero = torch.dist(averaged_embedding_half, averaged_embedding_zero)
    print(f"Distance between half and full: {distance_full}")
    print(f"Distance between half and ninety: {distance_ninety}")
    print(f"Distance between half and eighty: {distance_eighty}")
    print(f"Distance between half and seventy: {distance_seventy}")
    print(f"Distance between half and sixty: {distance_sixty}")
    print(f"Distance between half and fourty: {distance_fourty}")
    print(f"Distance between half and thirty: {distance_thirty}")
    print(f"Distance between half and twenty: {distance_twenty}")
    print(f"Distance between half and ten: {distance_ten}")
    print(f"Distance between half and zero: {distance_zero}")
    all_embeddings = np.stack([averaged_embedding_full.cpu().numpy(), averaged_embedding_ninety.cpu().numpy(), averaged_embedding_eighty.cpu().numpy(),
                               averaged_embedding_seventy.cpu().numpy(), averaged_embedding_sixty.cpu().numpy(), averaged_embedding_half.cpu().numpy(),
                               averaged_embedding_forty.cpu().numpy(), averaged_embedding_thirty.cpu().numpy(), averaged_embedding_twenty.cpu().numpy(),
                               averaged_embedding_ten.cpu().numpy(), averaged_embedding_zero.cpu().numpy()], axis=0)

    n_samples = all_embeddings.shape[0]
    perplexity_value = min(n_samples - 1, 30)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value)
    tsne_embeddings = tsne.fit_transform(all_embeddings)


    plt.scatter(tsne_embeddings[0, 0], tsne_embeddings[0, 1], color='blue', label='Full')
    plt.scatter(tsne_embeddings[1, 0], tsne_embeddings[1, 1], color='red', label='Ninety')
    plt.scatter(tsne_embeddings[2, 0], tsne_embeddings[2, 1], color='green', label='Eighty')
    plt.scatter(tsne_embeddings[3, 0], tsne_embeddings[3, 1], color='yellow', label='Seventy')
    plt.scatter(tsne_embeddings[4, 0], tsne_embeddings[4, 1], color='orange', label='Sixty')
    plt.scatter(tsne_embeddings[5, 0], tsne_embeddings[5, 1], color='purple', label='Half')
    plt.scatter(tsne_embeddings[6, 0], tsne_embeddings[6, 1], color='black', label='Fourty')
    plt.scatter(tsne_embeddings[7, 0], tsne_embeddings[7, 1], color='brown', label='Thirty')
    plt.scatter(tsne_embeddings[8, 0], tsne_embeddings[8, 1], color='pink', label='Twenty')
    plt.scatter(tsne_embeddings[9, 0], tsne_embeddings[9, 1], color='grey', label='Ten')
    plt.scatter(tsne_embeddings[10, 0], tsne_embeddings[10, 1], color='cyan', label='Zero')
plt.title('t-SNE Visualization')
plt.legend()
plt.savefig(f'embedding_drift_step_{token}.png')
plt.show()








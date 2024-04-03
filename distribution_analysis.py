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
model_name = "EleutherAI/pythia-160m-deduped-v0"
CHECKPOINT= 143000
model = GPTNeoXForCausalLM.from_pretrained(
  model_name,
  revision=f"step{CHECKPOINT}",
  cache_dir=f"./pythia-160m-deduped/step{CHECKPOINT}",
)
model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(
  model_name,
  revision=f"step{CHECKPOINT}",
  cache_dir=f"./pythia-160m-deduped/step{CHECKPOINT}",
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

df = pd.read_csv("generate_results/memorization_evals_160m-deduped-v0_32_48_143000.csv", index_col=0)
df_full_memorization = df[df['score'] == 1]
df_not_full_memorization = df[df['score'] == 0]
df_half_memorization = df[df['score'] == 0.5]


idx_full_memorization = df_full_memorization["idx"].tolist()
idx_not_full_memorization = df_not_full_memorization["idx"].tolist()
idx_half_memorization = df_half_memorization["idx"].tolist()

num_points = 100
highest_probability_memorized = logits_obtain(mmap_ds, model,  random.sample(idx_full_memorization,num_points), 32, 16)
highest_probability_unmemorized = logits_obtain(mmap_ds, model,  random.sample(idx_full_memorization,num_points), 32, 16)

plt.figure(figsize=(12, 8))  # 创建图像

# 使用不同的颜色和样式绘制两类数据
for i in range(100):
    hp_memorized_values = np.array([x[i].cpu() for x in highest_probability_memorized])
    hp_unmemorized_values = np.array([x[i].cpu() for x in highest_probability_unmemorized])

    changes_memorized = (hp_memorized_values - hp_memorized_values[0]) / hp_memorized_values[0] * 100  # 计算相对于初始值的百分比变化
    changes_unmemorized = (hp_unmemorized_values - hp_unmemorized_values[0]) / hp_unmemorized_values[
        0] * 100  # 计算相对于初始值的百分比变化

    plt.plot(range(16), changes_memorized, color='blue', linestyle='-', alpha=0.5)  # 类别1的样式
    plt.plot(range(16), changes_unmemorized, color='red', linestyle='--', alpha=0.5)  # 类别2的样式
# 创建图例来说明每个颜色和样式代表的类别
plt.plot([], [], color='blue', linestyle='-', label='Category 1')  # 类别1的图例
plt.plot([], [], color='red', linestyle='--', label='Category 2')  # 类别2的图例

plt.legend()  # 显示图例
plt.savefig(f'distribution.png')
plt.show()  # 显示图像

# plt.title('t-SNE Visualization')
# plt.legend()
# plt.savefig(f'tsne_visualization_{num_points}_{stragety}_{token}.png')
# plt.show()








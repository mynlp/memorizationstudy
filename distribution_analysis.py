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
model_name = "EleutherAI/pythia-410m-deduped-v0"
CHECKPOINT= 143000
model = GPTNeoXForCausalLM.from_pretrained(
  model_name,
  revision=f"step{CHECKPOINT}",
  cache_dir=f"./pythia-410m-deduped/step{CHECKPOINT}",
)
model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(
  model_name,
  revision=f"step{CHECKPOINT}",
  cache_dir=f"./pythia-410m-deduped/step{CHECKPOINT}",
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

df = pd.read_csv("generate_results/memorization_evals_410m-deduped-v0_32_48_143000.csv", index_col=0)
df_full_memorization = df[df['score'] == 1]
df_not_full_memorization = df[df['score'] == 0]
df_half_memorization = df[df['score'] == 0.5]


idx_full_memorization = df_full_memorization["idx"].tolist()
idx_not_full_memorization = df_not_full_memorization["idx"].tolist()
idx_half_memorization = df_half_memorization["idx"].tolist()

num_points = 500
highest_probability_memorized = logits_obtain(mmap_ds, model,  random.sample(idx_full_memorization,num_points), 32, 16)
highest_probability_unmemorized = logits_obtain(mmap_ds, model,  random.sample(idx_full_memorization,num_points), 32, 16)

plt.figure(figsize=(12, 8))  # 创建图像
# by-line plot
memorized_values = [np.array([x[i].cpu() for x in highest_probability_memorized])
                    for i in range(num_points)]
unmemorized_values = [np.array([x[i].cpu() for x in highest_probability_unmemorized])
                      for i in range(num_points)]
for values in memorized_values:
    plt.plot(range(16), values, color='blue', linestyle='-', alpha=0.1)  # 使用较低的透明度来避免图形过于拥挤

# 绘制未记忆化的每一条线
for values in unmemorized_values:
    plt.plot(range(16), values, color='red', linestyle='-', alpha=0.1)  # 使用较低的透明度

# 创建图例来说明每个颜色和样式代表的类别
plt.plot([], [], color='blue', linestyle='-', label='Memorized')  # 添加一个看不见的线作图例表示记忆化
plt.plot([], [], color='red', linestyle='-', label='Unmemorized')  # 添加一个看不见的线作图例表示未记忆化

plt.legend()  # 显示图例
plt.savefig(f'distribution_individual_lines.png')  # 保存图形
plt.show()  # 显示图形

# code for mean and std plot
# # 使用不同的颜色和样式绘制两类数据

#
# # 计算平均值和方差
# memorized_mean = np.mean(memorized_values, axis=0)
# memorized_std = np.std(memorized_values, axis=0)
# unmemorized_mean = np.mean(unmemorized_values, axis=0)
# unmemorized_std = np.std(unmemorized_values, axis=0)
#
# # 绘制平均值
# plt.plot(range(16), memorized_mean, color='blue', linestyle='-')
# plt.plot(range(16), unmemorized_mean, color='red', linestyle='-')
#
# # 添加阴影显示方差
# plt.fill_between(range(16), memorized_mean - memorized_std, memorized_mean + memorized_std,
#                  color='blue', alpha=0.2)
# plt.fill_between(range(16), unmemorized_mean - unmemorized_std, unmemorized_mean + unmemorized_std,
#                  color='red', alpha=0.2)
#
# # 创建图例来说明每个颜色和样式代表的类别
# plt.plot([], [], color='blue', linestyle='-', label='Category 1')  # 类别1的图例
# plt.plot([], [], color='red', linestyle='-', label='Category 2')  # 类别2的图例
#
# plt.legend()  # 显示图例
# plt.savefig(f'distribution.png')
# plt.show()  # 显示图像








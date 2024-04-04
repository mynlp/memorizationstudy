import pandas as pd
from transformers import GPTNeoXForCausalLM, AutoTokenizer,  AutoModelForCausalLM,  LogitsProcessorList, MinLengthLogitsProcessor, StoppingCriteriaList,  MaxLengthCriteria
from pythia.utils.mmap_dataset import MMapIndexedDataset
from transformers import GPTNeoXForCausalLM, AutoTokenizer
import torch
from utils import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random

# 定义一个简单的函数来计算移动平均
def moving_average(data, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(data, window, 'same')

random.seed(42)
model_name = "EleutherAI/pythia-70m-deduped-v0"
CHECKPOINT= 143000
window_size = 3
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

df = pd.read_csv("generate_results/memorization_evals_70m-deduped-v0_32_96_143000.csv", index_col=0)
df_full_memorization = df[df['score'] == 1]
df_not_full_memorization = df[df['score'] == 0]
df_half_memorization = df[df['score'] == 0.5]


idx_full_memorization = df_full_memorization["idx"].tolist()
idx_not_full_memorization = df_not_full_memorization["idx"].tolist()
idx_half_memorization = df_half_memorization["idx"].tolist()

num_points = 500
highest_probability_memorized = logits_obtain(mmap_ds, model,  random.sample(idx_full_memorization,num_points), 32, 64)
highest_probability_unmemorized = logits_obtain(mmap_ds, model,  random.sample(idx_full_memorization,num_points), 32, 64)

plt.figure(figsize=(12, 8))  # 创建图像
# by-line plot
memorized_values = [np.array([x[i].cpu() for x in highest_probability_memorized])
                    for i in range(num_points)]
memorized_mean = np.mean(memorized_values, axis=0)
unmemorized_values = [np.array([x[i].cpu() for x in highest_probability_unmemorized])
                      for i in range(num_points)]
unmemorized_mean = np.mean(unmemorized_values, axis=0)
memorized_rolling_means = [moving_average(values, window_size) for values in memorized_values]
unmemorized_rolling_means = [moving_average(values, window_size) for values in unmemorized_values]
# 用低透明度绘制每一条记忆化数据的线
for values in memorized_rolling_means:
    plt.plot(len(memorized_rolling_means[0]), values, color='red', linestyle='-', alpha=0.1)

# 用低透明度绘制每一条未记忆化数据的线
for values in unmemorized_rolling_means:
    plt.plot(len(memorized_rolling_means[0]), values, color='blue', linestyle='-', alpha=0.1)

# 绘制记忆化和未记忆化数据的平均线
plt.plot(len(memorized_mean[0]), memorized_mean, color='darkred', linestyle='-', linewidth=2, label='Average Memorized')
plt.plot(len(unmemorized_mean[0]), unmemorized_mean, color='darkblue', linestyle='-', linewidth=2, label='Average Unmemorized')

# 创建图例来说明每个颜色和样式代表的类别
# 这里解释了平均线的颜色和透明度较低的每条线
plt.plot([], [], color='red', linestyle='-', alpha=0.1, label='Individual Memorized Instances')
plt.plot([], [], color='blue', linestyle='-', alpha=0.1, label='Individual Unmemorized Instances')
plt.plot([], [], color='darkred', linestyle='-', linewidth=2, label='Average Memorized')
plt.plot([], [], color='darkblue', linestyle='-', linewidth=2, label='Average Unmemorized')

# 添加标题和坐标轴标签
plt.title('Comparison of Memorized and Unmemorized Data Over Time')
plt.xlabel('Time Point')
plt.ylabel('Data Value')

plt.legend()
plt.savefig('distribution_individual_lines.png')
plt.show()

# code for mean and std plot
# # 使用不同的颜色和样式绘制两类数据

#
# # 计算平均值和方差
#memorized_mean = np.mean(memorized_values, axis=0)
# memorized_std = np.std(memorized_values, axis=0)
#unmemorized_mean = np.mean(unmemorized_values, axis=0)
# unmemorized_std = np.std(unmemorized_values, axis=0)
#
# # 绘制平均值

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








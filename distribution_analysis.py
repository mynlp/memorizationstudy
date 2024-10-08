import pandas as pd
from transformers import GPTNeoXForCausalLM, AutoTokenizer,  AutoModelForCausalLM,  LogitsProcessorList, MinLengthLogitsProcessor, StoppingCriteriaList,  MaxLengthCriteria
#from pythia.utils.mmap_dataset import MMapIndexedDataset
from transformers import GPTNeoXForCausalLM, AutoTokenizer
import torch
from utils import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
import pickle
# 定义一个简单的函数来计算移动平均
def moving_average(data, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(data, window, 'same')

prefix = 'deduped_merge/document.bin'
print(prefix)
buff_size = 2049*1024*2
print("Building dataset")
#mmap_ds = MMapIndexedDataset(prefix, skip_warmup=True)
random.seed(42)
memorized_entropy_value = []
half_memorized_entropy_value = []
unmemorized_entropy_value = []
#model_size_list = ["70m","410m", "1b", "2.8b", "6.9b", "12b"]
model_size_list = ["70m","410m", "1b", "2.8b", "6.9b", "12b"]
# for model_size in model_size_list:
#     model_name = f"EleutherAI/pythia-{model_size}-deduped-v0"
#     CHECKPOINT = 143000
#     window_size = 5
#     context = 32
#     continuation = 16
#     model = GPTNeoXForCausalLM.from_pretrained(
#       model_name,
#       revision=f"step{CHECKPOINT}",
#       cache_dir=f"./pythia-{model_size}-deduped/step{CHECKPOINT}",
#     ).bfloat16().eval()
#     model = model.to_bettertransformer()
#     #model = model.cuda()
#     tokenizer = AutoTokenizer.from_pretrained(
#       model_name,
#       revision=f"step{CHECKPOINT}",
#       cache_dir=f"./pythia-{model_size}-deduped/step{CHECKPOINT}",
#     )
#     model.generation_config.pad_token_id = model.generation_config.eos_token_id
#     #model.generation_config.output_hidden_states = True
#     #model.generation_config.output_attentions = True
#     model.generation_config.output_scores = True
#     model.generation_config.return_dict_in_generate = True
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     device_ids = list(range(torch.cuda.device_count()))
#     # 将模型加载到所有 GPU 上
#     model = torch.nn.DataParallel(model)
#     model = model.to('cuda')
#
#     df = pd.read_csv(f"generate_results/memorization_evals_{model_size}-deduped-v0_{context}_{context+continuation}_143000.csv", index_col=0)
#     df_full_memorization = df[df['score'] == 1]
#     df_not_full_memorization = df[df['score'] == 0]
#     df_half_memorization = df[df['score'] == 0.5]
#
#
#     idx_full_memorization = df_full_memorization["idx"].tolist()
#     idx_not_full_memorization = df_not_full_memorization["idx"].tolist()
#     idx_half_memorization = df_half_memorization["idx"].tolist()
#
#     num_points = 1000
#     highest_probability_memorized = logits_obtain(mmap_ds, model,  random.sample(idx_full_memorization,num_points), context, continuation)
#     highest_probability_half_memorized = logits_obtain(mmap_ds, model,  random.sample(idx_half_memorization,num_points), context, continuation)
#     highest_probability_unmemorized = logits_obtain(mmap_ds, model,  random.sample(idx_not_full_memorization,num_points), context, continuation)
#
#     plt.figure(figsize=(12, 8))  # 创建图像
#     # by-line plot
#     memorized_values = [np.array([x[i].cpu() for x in highest_probability_memorized])
#                         for i in range(num_points)]
#     memorized_mean = np.mean(memorized_values, axis=0)
#     half_memorized_values = [np.array([x[i].cpu() for x in highest_probability_half_memorized]
#                                       ) for i in range(num_points)]
#     half_memorized_mean = np.mean(half_memorized_values, axis=0)
#     unmemorized_values = [np.array([x[i].cpu() for x in highest_probability_unmemorized])
#                           for i in range(num_points)]
#     unmemorized_mean = np.mean(unmemorized_values, axis=0)
#
#     memorized_entropy_value.append(memorized_mean)
#     half_memorized_entropy_value.append(half_memorized_mean)
#     unmemorized_entropy_value.append(unmemorized_mean)
#
#     memorized_rolling_means = [moving_average(values, window_size) for values in memorized_values]
#     half_memorized_rolling_means = [moving_average(values, window_size) for values in half_memorized_values]
#     unmemorized_rolling_means = [moving_average(values, window_size) for values in unmemorized_values]
#     # 用低透明度绘制每一条记忆化数据的线
#     for values in memorized_rolling_means:
#         plt.plot(range(len(memorized_rolling_means[0])), values, color='red', linestyle='-', alpha=0.1)
#
#     for values in half_memorized_rolling_means:
#         plt.plot(range(len(memorized_rolling_means[0])), values, color='green', linestyle='-', alpha=0.1)
#
#     # 用低透明度绘制每一条未记忆化数据的线
#     for values in unmemorized_rolling_means:
#         plt.plot(range(len(memorized_rolling_means[0])), values, color='blue', linestyle='-', alpha=0.1)
#
#     # 绘制记忆化和未记忆化数据的平均线
#     plt.plot(range(len(memorized_mean)), memorized_mean, color='darkred', linestyle='-', linewidth=2, label='Average Memorized')
#     plt.plot(range(len(half_memorized_mean)), half_memorized_mean, color='darkgreen', linestyle='-', linewidth=2, label='Average Half Memorized')
#     plt.plot(range(len(unmemorized_mean)), unmemorized_mean, color='darkblue', linestyle='-', linewidth=2, label='Average Unmemorized')
#
#     # 创建图例来说明每个颜色和样式代表的类别
#     # 这里解释了平均线的颜色和透明度较低的每条线
#     plt.plot([], [], color='red', linestyle='-', alpha=0.1, label='Individual Memorized Instances')
#     plt.plot([], [], color='blue', linestyle='-', alpha=0.1, label='Individual Unmemorized Instances')
#     plt.plot([], [], color='green', linestyle='-', alpha=0.1, label='Individual Half Memorized Instances')
#     plt.plot([], [], color='darkred', linestyle='-', linewidth=2, label='Average Memorized')
#     plt.plot([], [], color='darkgreen', linestyle='-', linewidth=2, label='Average Half Memorized')
#     plt.plot([], [], color='darkblue', linestyle='-', linewidth=2, label='Average Unmemorized')
#     # 添加标题和坐标轴标签
#     plt.title('Comparison of Memorized and Unmemorized Data Over Time')
#     plt.xlabel('Time Point')
#     plt.ylabel('Data Value')
#     plt.legend()
#     plt.savefig(f'distribution_individual_lines_{context}_{continuation}_{model_size}.png')
#     plt.show()
# f = open("memorized_entropy_value.pkl", "wb")
# pickle.dump(memorized_entropy_value, f)
# f.close()
# f = open("half_memorized_entropy_value.pkl", "wb")
# pickle.dump(half_memorized_entropy_value, f)
# f.close()
# f = open("unmemorized_entropy_value.pkl", "wb")
# pickle.dump(unmemorized_entropy_value, f)
# f.close()

f = open("memorized_entropy_value.pkl", "rb")
memorized_entropy_value = pickle.load(f)
f.close()
f = open("half_memorized_entropy_value.pkl", "rb")
half_memorized_entropy_value = pickle.load(f)
f.close()
f = open("unmemorized_entropy_value.pkl", "rb")
unmemorized_entropy_value = pickle.load(f)
f.close()

# plt.figure(figsize=(12, 8))
# plt.plot(model_size_list, [x[0] for x in memorized_entropy_value], color='red', label=f'inital_token_memorized')
# plt.plot(model_size_list, [x[0] for x in unmemorized_entropy_value], color='blue', label=f'inital_token_unmemorized')
# plt.plot(model_size_list, [x[-1] for x in memorized_entropy_value], color='darkred', label=f'last_token_memorized')
# plt.plot(model_size_list, [x[-1] for x in unmemorized_entropy_value],color='darkblue', label=f'last_token_unmemorized')
# plt.title('Entropy at Initial and Last Token for Memorized and Unmemorized Data')
# plt.xlabel('Model Size')
# plt.ylabel('Entropy')
# plt.legend()
# plt.savefig(f'entropy_across_size.png')
# plt.show()
context = 32
continuation = 16
plt.figure(figsize=(12, 8))
memorized_entropy_values = [memorized_entropy_value[i][19:] for i in range(len(model_size_list))]
half_memorized_entropy_values = [half_memorized_entropy_value[i][19:] for i in range(len(model_size_list))]
unmemorized_entropy_values = [unmemorized_entropy_value[i][19:] for i in range(len(model_size_list))]
colors = ['red', 'green', 'blue', 'darkred', 'darkgreen', 'darkblue', 'purple', 'orange', 'yellow',
          'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'lightblue', 'lightgreen', 'lightyellow',
          'lightgray', 'darkgray', 'lavender', 'turquoise', 'teal', 'peachpuff', 'navy', 'salmon',
          'gold', 'black', 'beige', 'lime', 'coral', 'plum', 'tan', 'skyblue', 'aquamarine',
          'slategray', 'orchid', 'hotpink', 'mediumspringgreen', 'khaki', 'seagreen',
          'steelblue', 'powderblue', 'indigo', 'burlywood', 'darkmagenta', 'midnightblue',
          'royalblue', 'mediumblue', 'palegreen', 'peru', 'lightpink', 'crimson',
          'white', 'azure', 'oldlace', 'mintcream', 'linen', 'aliceblue', 'ghostwhite',
          'honeydew', 'floralwhite', 'cornsilk', 'snow', 'seashell', 'ivory', 'lemonchiffon']
linestyle_list = ['-', '--', '-.', ':', '-', '--']
x_values = range(20, context + continuation)
for i in range(len(model_size_list)):
    plt.plot(x_values, memorized_entropy_values[i], color=colors[3 * i], label=f'{model_size_list[i]}_memorized', linestyle=linestyle_list[0])
    plt.plot(x_values, half_memorized_entropy_values[i], color=colors[3 * i+1], label=f'{model_size_list[i]}_half_memorized', linestyle=linestyle_list[1])
    plt.plot(x_values, unmemorized_entropy_values[i], color=colors[3 * i+2], label=f'{model_size_list[i]}_unmemorized', linestyle=linestyle_list[3])
min_memorized = np.min(memorized_entropy_values, axis=0)
max_memorized = np.max(memorized_entropy_values, axis=0)
min_half_memorized = np.min(half_memorized_entropy_values, axis=0)
max_half_memorized = np.max(half_memorized_entropy_values, axis=0)
min_unmemorized = np.min(unmemorized_entropy_values, axis=0)
max_unmemorized = np.max(unmemorized_entropy_values, axis=0)
plt.fill_between(range(20, context + continuation), min_memorized, max_memorized, color="blue", alpha=0.35, label='memorized')
plt.fill_between(range(20, context + continuation), min_half_memorized, max_half_memorized, color="orange", alpha=0.35, label='half memorized')
plt.fill_between(range(20, context + continuation), min_unmemorized, max_unmemorized, color="green", alpha=0.35, label='unmemorized')
plt.axvline(x=31, color='red', linestyle='--')
plt.text(31-2.5, 4, 'Context\nEnd Point', rotation=0, size=14)
plt.axvline(x=32, color='blue', linestyle='--')
plt.text(32+0.3, 4, 'Decoding\nStart Point', rotation=0, size=14)
plt.axvline(x=40, color='black', linestyle='--')
plt.text(40-2.7, 4, 'Decoding\nHalf Point', rotation=0, size=14)
plt.legend(loc='upper left', fontsize='15', handlelength=1.5, handletextpad=0.5)
plt.title('Entropy at Each Token for Memorized and Unmemorized Data',  fontsize=14)
plt.xlabel('Token Position', fontsize=14)
plt.ylabel('Entropy', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f'entropy_across_steps.png', bbox_inches='tight', dpi=600)
plt.show()









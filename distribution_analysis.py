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
f = open("results/memorized_entropy_value.pkl", "rb")
memorized_entropy_value = pickle.load(f)
f.close()
f = open("results/half_memorized_entropy_value.pkl", "rb")
half_memorized_entropy_value = pickle.load(f)
f.close()
f = open("results/unmemorized_entropy_value.pkl", "rb")
unmemorized_entropy_value = pickle.load(f)
f.close()

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









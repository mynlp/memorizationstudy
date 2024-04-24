import pandas as pd
from transformers import GPTNeoXForCausalLM, AutoTokenizer,  AutoModelForCausalLM,  LogitsProcessorList, MinLengthLogitsProcessor, StoppingCriteriaList,  MaxLengthCriteria
from pythia.utils.mmap_dataset import MMapIndexedDataset
from transformers import GPTNeoXForCausalLM, AutoTokenizer
import torch
from utils import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
import plotly.graph_objects as go
import pandas as pd
import plotly.io as pio

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
df_large = pd.read_csv(f"generate_results/memorization_evals_{large_model_size}-deduped-v0_{context}_{context+continuation}_143000.csv", index_col=0)
df_small_memorized = df_small[df_small["score"] == 1]
df_large_memorized = df_large[df_large["score"] == 1]
scores = pd.concat([df_small["score"], df_large["score"]]).unique()
source = []
target = []
value = []
label = []
for i, score in enumerate(scores):
    # 在每个数据框中找出所有这个score的部分
    df_small_score = df_small[df_small["score"] == score]
    df_large_score = df_large[df_large["score"] == score]

    # 计算这个得分的数量并添加到source、target、value
    score_count = len(df_small_score)
    source += [i] * score_count
    target += [i + len(scores)] * score_count
    value += [score_count]

    # 添加label
    label += [f"Small-Score {score}", f"Large-Score {score}"]

# 现在source、target、value、label列表已经包含了所有得分的信息
# 接下来创建sankey图

fig = go.Figure(data=[
    go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=label,
            color="blue"
        ),
        link=dict(
            source=source,
            target=target,
            value=value
        ),
    )
])
fig.pio.write_image(fig, 'sankey_diagram.png')
fig.show()


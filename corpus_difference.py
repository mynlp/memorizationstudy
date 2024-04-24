import pandas as pd
from pythia.utils.mmap_dataset import MMapIndexedDataset
from utils import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
import plotly.graph_objects as go
import pandas as pd
import plotly.io as pio
from tqdm import tqdm

random.seed(42)
small_model_size = "70m"
large_model_size = "410m"
context = 32
continuation = 16
prefix = 'deduped_merge/document.bin'
print(prefix)
buff_size = 2049*1024*2
print("Building dataset")
#mmap_ds = MMapIndexedDataset(prefix, skip_warmup=True)

df_small = pd.read_csv(f"generate_results/memorization_evals_{small_model_size}-deduped-v0_{context}_{context+continuation}_143000.csv", index_col=0)
df_large = pd.read_csv(f"generate_results/memorization_evals_{large_model_size}-deduped-v0_{context}_{context+continuation}_143000.csv", index_col=0)
df_small_memorized = df_small[df_small["score"] == 1]
df_large_memorized = df_large[df_large["score"] == 1]
scores = pd.concat([df_small["score"], df_large["score"]]).unique()
df_small["idx"] = df_small.index
df_large["idx"] = df_large.index

# 连接 df_small 和 df_large
df = pd.merge(df_small, df_large, on="idx", suffixes=("_small", "_large"))

# 创建转移矩阵
transition_matrix = pd.crosstab(df["score_small"], df["score_large"])

# 为Sankey图生成必要的数据
label = list(transition_matrix.index.astype(str)) + list(transition_matrix.columns.astype(str))
source = []
target = []
value = []

for r, row in tqdm(enumerate(transition_matrix.index)):
    for c, col in enumerate(transition_matrix.columns):
        source.append(r)
        target.append(c + len(transition_matrix.index))
        value.append(transition_matrix.at[row, col])

# 创建Sankey图
fig = go.Figure(data=[go.Sankey(
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
        value=value,
    ))])
pio.write_image(fig, 'sankey_diagram.png')
fig.show()


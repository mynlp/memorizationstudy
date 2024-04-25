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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def redefine_score(score):
    if score <= 0.2:
        return 'very low'
    elif score <= 0.4:
        return 'low'
    elif score <= 0.6:
        return 'medium'
    elif score <= 0.8:
        return 'high'
    else:
        return 'very high'

random.seed(42)
small_model_size = "410m"
large_model_size = "2.8b"
extra_large_model_size = "12b"
context = 32
continuation = 16
prefix = 'deduped_merge/document.bin'
print(prefix)
buff_size = 2049*1024*2
print("Building dataset")
#mmap_ds = MMapIndexedDataset(prefix, skip_warmup=True)

df_small = pd.read_csv(f"generate_results/memorization_evals_{small_model_size}-deduped-v0_{context}_{context+continuation}_143000.csv", index_col=0)
df_large = pd.read_csv(f"generate_results/memorization_evals_{large_model_size}-deduped-v0_{context}_{context+continuation}_143000.csv", index_col=0)
df_extra_large = pd.read_csv(f"generate_results/memorization_evals_{extra_large_model_size}-deduped-v0_{context}_{context+continuation}_143000.csv", index_col=0)
# df_small_memorized = df_small[df_small["score"] == 1]
# df_large_memorized = df_large[df_large["score"] == 1]
# df_extra_large_memorized = df_extra_large[df_extra_large["score"] == 1]
scores = pd.concat([df_small["score"], df_large["score"]]).unique()

df_small['score'] = df_small['score'].apply(redefine_score)
df_large['score'] = df_large['score'].apply(redefine_score)
df_extra_large["score"] = df_extra_large["score"].apply(redefine_score)

df_small["idx"] = df_small.index
df_large["idx"] = df_large.index
df_extra_large["idx"] = df_extra_large.index
# 连接 df_small 和 df_large
df = pd.merge(df_small, df_large, on="idx", suffixes=("_small", "_large"))
df_new = pd.merge(df_large, df_extra_large, left_on="idx", right_on="idx", suffixes=("_large", "_extra_large"))
# 创建转移矩阵
transition_matrix = pd.crosstab(df_small["score"], df_large["score"])
transition_matrix_value = transition_matrix.values
transition_matrix_extra_large = pd.crosstab(df_large["score"], df_extra_large["score"])
transition_matrix_value_extra_large = transition_matrix_extra_large.values

transition_prob_matrix_small_large = transition_matrix_value / transition_matrix_value.sum(axis=1,keepdims=True)
transition_prob_matrix_large_extra_large = transition_matrix_value_extra_large / transition_matrix_value_extra_large.sum(
    axis=1, keepdims=True)

plt.figure(figsize=(20, 8))

plt.subplot(1, 2, 1)
sns.heatmap(transition_prob_matrix_small_large, annot=True, cmap="viridis", fmt=".3f",
            xticklabels=df_large["score"].unique(),
            yticklabels=df_small["score"].unique())
plt.title('Transition Probability Matrix Small to Large - Heatmap')
plt.xlabel('df_large')
plt.ylabel('df_small')

plt.subplot(1, 2, 2)
sns.heatmap(transition_prob_matrix_large_extra_large, annot=True, cmap="viridis", fmt=".3f",
            xticklabels=df_extra_large["score"].unique(),
            yticklabels=df_large["score"].unique())
plt.title('Transition Probability Matrix Large to Extra_Large - Heatmap')
plt.xlabel('df_extra_large')
plt.ylabel('df_large')
plt.savefig("transition_matrix.png")
plt.show()
# label = list(transition_matrix.index.astype(str)) + list(transition_matrix.columns.astype(str))
#
# transition_matrix_large_extra_large = pd.crosstab(df_large["score"], df_extra_large["score"])
# label_large_extra_large = list(transition_matrix_large_extra_large.index.astype(str)) + list(
#     transition_matrix_large_extra_large.columns.astype(str))
# # 为Sankey图生成必要的数据
# #label = list(transition_matrix.index.astype(str)) + list(transition_matrix.columns.astype(str))
# source = []
# target = []
# value = []
#
# for r, row in tqdm(enumerate(transition_matrix.index)):
#     for c, col in enumerate(transition_matrix.columns):
#         source.append(r)
#         target.append(c + len(transition_matrix.index))
#         value.append(transition_matrix.at[row, col])
#
# for r, row in tqdm(enumerate(transition_matrix_large_extra_large.index)):
#     for c, col in enumerate(transition_matrix_large_extra_large.columns):
#         source.append(r + len(transition_matrix.index))  # 索引偏移，以便于连接到上一层
#         target.append(c + len(transition_matrix.index) + len(transition_matrix_large_extra_large.index))  # 索引偏移
#         value.append(transition_matrix_large_extra_large.at[row, col])
#
# label = label + label_large_extra_large
# color = ["blue", "red", "green", "yellow", "purple"] * (len(label) // 5 + 1)
#
# # 创建Sankey图
# fig = go.Figure(data=[go.Sankey(
#     node=dict(
#         pad=5,
#         thickness=20,
#         line=dict(color="black", width=0.5),
#         label=label,
#         color=color[:len(label)],  # 确保颜色列表和标签列表长度一致
#     ),
#     link=dict(
#         source=source,
#         target=target,
#         value=value,
#         color=["rgba(31,119,180,0.8)", "rgba(255,127,14,0.8)",
#                "rgba(44,160,44,0.8)", "rgba(214,39,40,0.8)",
#                "rgba(148,103,189,0.8)", "rgba(140,86,75,0.8)",
#                "rgba(227,119,194,0.8)", "rgba(127,127,127,0.8)",
#                "rgba(188,189,34,0.8)"] * (len(value) // 9 + 1)  # 确保颜色列表和值列表长度一致
#     ))])
# # 更新图设定
# fig.update_layout(
#     title_text="Sankey Diagram from small to large to extra_large",  # 修改标题
#     font=dict(
#         family="Times New Roman",  # 更改字体类型
#         size=15,
#     ),
# )
# pio.write_image(fig, 'sankey_diagram.png', dpi=800)
# #fig.show()

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
# def redefine_score(score):
#     if score <= 0.2:
#         return 'very low'
#     elif score <= 0.4:
#         return 'low'
#     elif score <= 0.6:
#         return 'medium'
#     elif score <= 0.8:
#         return 'high'
#     else:
#         return 'very high'


def redefine_score(score):
    if score <= 0.1:
        return 'very low'
    elif score <= 0.2:
        return 'low'
    elif score <= 0.3:
        return 'low medium'
    elif score <= 0.4:
        return 'medium'
    elif score <= 0.5:
        return 'high medium'
    elif score <= 0.6:
        return 'high'
    elif score <= 0.7:
        return 'very high'
    elif score <= 0.8:
        return 'extremely high'
    elif score <= 0.9:
        return 'almost perfect'
    else:
        return 'perfect'

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
# df = pd.merge(df_small, df_large, on="idx", suffixes=("_small", "_large"))
# df_new = pd.merge(df_large, df_extra_large, left_on="idx", right_on="idx", suffixes=("_large", "_extra_large"))
# # 创建转移矩阵
# transition_matrix = pd.crosstab(df_small["score"], df_large["score"])
# transition_matrix_value = transition_matrix.values
# transition_matrix_extra_large = pd.crosstab(df_large["score"], df_extra_large["score"])
# transition_matrix_value_extra_large = transition_matrix_extra_large.values
#
# transition_prob_matrix_small_large = transition_matrix_value / transition_matrix_value.sum(axis=1,keepdims=True)
# transition_prob_matrix_large_extra_large = transition_matrix_value_extra_large / transition_matrix_value_extra_large.sum(
#     axis=1, keepdims=True)
#
# fig, axs = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={'wspace': 0.2})
# plt.rcParams.update({'font.size': 14})
# cbar_ax = fig.add_axes([.91, .12, .03, .76])
# plt.subplot(1, 2, 1)
# sns.heatmap(transition_prob_matrix_small_large, annot=True, cmap="viridis", fmt=".3f",
#             xticklabels=df_large["score"].unique(),
#             yticklabels=df_small["score"].unique(), annot_kws={"size": 16}, ax=axs[0], cbar=False)
# axs[0].set_title('Transition Matrix 410m to 2.8b')
# axs[0].set_xlabel('2.8b Model')
# axs[0].set_ylabel('410m Model')
# sns.heatmap(transition_prob_matrix_large_extra_large, annot=True, cmap="viridis", fmt=".3f",
#             xticklabels=df_extra_large["score"].unique(),
#             yticklabels=df_large["score"].unique(), annot_kws={"size": 16}, ax=axs[1],
#             cbar_ax=cbar_ax)
# axs[1].set_title('Transition Matrix 2.8b to 12b')
# axs[1].set_xlabel('12b Model')
# axs[1].set_ylabel('2.8b Model')
# plt.savefig('transition_matrix.png', bbox_inches='tight', dpi=600)
# plt.show()

df_reverse_1 = pd.merge(df_extra_large, df_large, on="idx", suffixes=("_extra_large", "_large"))
transition_matrix_reverse_1 = pd.crosstab(df_reverse_1["score_extra_large"], df_reverse_1["score_large"])
transition_matrix_value_reverse_1 = transition_matrix_reverse_1.values
transition_prob_matrix_reverse_1 = transition_matrix_value_reverse_1 / transition_matrix_value_reverse_1.sum(axis=1, keepdims=True)

# From 2.8b to 410m
df_reverse_2 = pd.merge(df_large, df_small, on="idx", suffixes=("_large", "_small"))
transition_matrix_reverse_2 = pd.crosstab(df_reverse_2["score_large"], df_reverse_2["score_small"])
transition_matrix_value_reverse_2 = transition_matrix_reverse_2.values
transition_prob_matrix_reverse_2 = transition_matrix_value_reverse_2 / transition_matrix_value_reverse_2.sum(axis=1, keepdims=True)

# 绘制反向转移矩阵的热力图
fig, axs = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={'wspace': 0.2})
plt.rcParams.update({'font.size': 14})

cbar_ax = fig.add_axes([.91, .12, .03, .76])
plt.subplot(1, 2, 1)
sns.heatmap(transition_prob_matrix_reverse_1, annot=True, cmap="viridis", fmt=".3f",
            xticklabels=df_large["score"].unique(), yticklabels=df_extra_large["score"].unique(),
            annot_kws={"size": 16}, ax=axs[0], cbar=False)
axs[0].set_title('Transition Matrix 12b to 2.8b')
axs[0].set_xlabel('2.8b Model')
axs[0].set_ylabel('12b Model')

sns.heatmap(transition_prob_matrix_reverse_2, annot=True, cmap="viridis", fmt=".3f",
            xticklabels=df_small["score"].unique(), yticklabels=df_large["score"].unique(),
            annot_kws={"size": 16}, ax=axs[1], cbar_ax=cbar_ax)
axs[1].set_title('Transition Matrix 2.8b to 410m')
axs[1].set_xlabel('410m Model')
axs[1].set_ylabel('2.8b Model')
print(transition_prob_matrix_reverse_1)
print(transition_prob_matrix_reverse_2)
plt.savefig('reverse_transition_matrix.png', bbox_inches='tight', dpi=600)
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

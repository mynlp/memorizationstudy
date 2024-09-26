import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 定义用于重新定义评分的函数
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

# 设定随机种子（如果需要）
np.random.seed(42)

# 定义模型大小
small_model_size = "410m"
large_model_size = "2.8b"
extra_large_model_size = "12b"

# 读取数据集（请根据实际文件路径和名称替换下面的路径）
context = 32
continuation = 16
df_small = pd.read_csv(f"generate_results/memorization_evals_{small_model_size}-deduped-v0_{context}_{context+continuation}_143000.csv", index_col=0)
df_large = pd.read_csv(f"generate_results/memorization_evals_{large_model_size}-deduped-v0_{context}_{context+continuation}_143000.csv", index_col=0)
df_extra_large = pd.read_csv(f"generate_results/memorization_evals_{extra_large_model_size}-deduped-v0_{context}_{context+continuation}_143000.csv", index_col=0)

# 应用评分重新定义函数
df_small['score'] = df_small['score'].apply(redefine_score)
df_large['score'] = df_large['score'].apply(redefine_score)
df_extra_large['score'] = df_extra_large['score'].apply(redefine_score)

# 确保每个DataFrame都有'idx'列，用于合并
# df_small = df_small.reset_index().rename(columns={'index': 'idx'})
# df_large = df_large.reset_index().rename(columns={'index': 'idx'})
# df_extra_large = df_extra_large.reset_index().rename(columns={'index': 'idx'})

# 定义评分标签
score_labels = ['very low', 'low', 'medium', 'high', 'very high']

### 计算从小模型到大模型的转移矩阵 P(score_large | score_small) ###

# 合并 df_small 和 df_large
df_small_large = pd.merge(df_small[['idx', 'score']], df_large[['idx', 'score']], on='idx', suffixes=('_small', '_large'))

# 创建转移矩阵：行是小模型评分，列是大模型评分
transition_matrix_small_large = pd.crosstab(df_small_large['score_small'], df_small_large['score_large'])
transition_matrix_small_large = transition_matrix_small_large.reindex(index=score_labels, columns=score_labels, fill_value=0)

# 计算条件概率矩阵 P(score_large | score_small)
transition_prob_matrix_small_large = transition_matrix_small_large.div(transition_matrix_small_large.sum(axis=1), axis=0).fillna(0)

### 计算从大模型到更大模型的转移矩阵 P(score_extra_large | score_large) ###

# 合并 df_large 和 df_extra_large
df_large_extra_large = pd.merge(df_large[['idx', 'score']], df_extra_large[['idx', 'score']], on='idx', suffixes=('_large', '_extra_large'))

# 创建转移矩阵：行是大模型评分，列是更大模型评分
transition_matrix_large_extra_large = pd.crosstab(df_large_extra_large['score_large'], df_large_extra_large['score_extra_large'])
transition_matrix_large_extra_large = transition_matrix_large_extra_large.reindex(index=score_labels, columns=score_labels, fill_value=0)

# 计算条件概率矩阵 P(score_extra_large | score_large)
transition_prob_matrix_large_extra_large = transition_matrix_large_extra_large.div(transition_matrix_large_extra_large.sum(axis=1), axis=0).fillna(0)

### 计算从更大模型到大模型的逆转移矩阵 P(score_large | score_extra_large) ###

# 合并 df_extra_large 和 df_large
df_extra_large_large = pd.merge(df_extra_large[['idx', 'score']], df_large[['idx', 'score']], on='idx', suffixes=('_extra_large', '_large'))

# 创建逆转移矩阵：行是更大模型评分，列是大模型评分
transition_matrix_extra_large_large = pd.crosstab(df_extra_large_large['score_extra_large'], df_extra_large_large['score_large'])
transition_matrix_extra_large_large = transition_matrix_extra_large_large.reindex(index=score_labels, columns=score_labels, fill_value=0)

# 计算条件概率矩阵 P(score_large | score_extra_large)
transition_prob_matrix_extra_large_large = transition_matrix_extra_large_large.div(transition_matrix_extra_large_large.sum(axis=1), axis=0).fillna(0)

### 计算从大模型到小模型的逆转移矩阵 P(score_small | score_large) ###

# 合并 df_large 和 df_small
df_large_small = pd.merge(df_large[['idx', 'score']], df_small[['idx', 'score']], on='idx', suffixes=('_large', '_small'))

# 创建逆转移矩阵：行是大模型评分，列是小模型评分
transition_matrix_large_small = pd.crosstab(df_large_small['score_large'], df_large_small['score_small'])
transition_matrix_large_small = transition_matrix_large_small.reindex(index=score_labels, columns=score_labels, fill_value=0)

# 计算条件概率矩阵 P(score_small | score_large)
transition_prob_matrix_large_small = transition_matrix_large_small.div(transition_matrix_large_small.sum(axis=1), axis=0).fillna(0)

### 绘制所有四个转移矩阵的热力图 ###

# 设置绘图风格和大小
#sns.set(style='whitegrid', font_scale=1.2)
fig, axs = plt.subplots(1, 4, figsize=(32, 8), gridspec_kw={'wspace': 0.4})
plt.rcParams.update({'font.size': 16})
cbar_ax = fig.add_axes([.92, .15, .02, .7])
for ax in axs:
    ax.set_aspect('equal')
# 从小模型到大模型的转移矩阵
sns.heatmap(transition_prob_matrix_small_large, annot=True, cmap="viridis", fmt=".3f",
            xticklabels=score_labels, yticklabels=score_labels,  annot_kws={"size": 16}, ax=axs[0], cbar=False, square=True)
axs[0].set_title('Transition Matrix \n 410m to 2.8b')
axs[0].set_xlabel('2.8b Model')
axs[0].set_ylabel('410m Model')

# 从大模型到更大模型的转移矩阵
sns.heatmap(transition_prob_matrix_large_extra_large, annot=True, cmap="viridis", fmt=".3f",
            xticklabels=score_labels, yticklabels=score_labels, ax=axs[1], annot_kws={"size": 16}, cbar=False, square=True)
axs[1].set_title('Transition Matrix \n 2.8b to 12b')
axs[1].set_xlabel('12b Model')
axs[1].set_ylabel('2.8b Model')

# 从更大模型到大模型的逆转移矩阵
sns.heatmap(transition_prob_matrix_extra_large_large, annot=True, cmap="viridis", fmt=".3f",
            xticklabels=score_labels, yticklabels=score_labels, ax=axs[2],  annot_kws={"size": 16}, cbar=False, square=True)
axs[2].set_title('Inverse Transition Matrix\n 12b to 2.8b')
axs[2].set_xlabel('2.8b Model')
axs[2].set_ylabel('12b Model')

# 从大模型到小模型的逆转移矩阵
sns.heatmap(transition_prob_matrix_large_small, annot=True, cmap="viridis", fmt=".3f",
            xticklabels=score_labels, yticklabels=score_labels, ax=axs[3], annot_kws={"size": 16}, cbar_ax=cbar_ax, square=True)
axs[3].set_title('Inverse Transition Matrix\n 2.8b to 410m')
axs[3].set_xlabel('410m Model')
axs[3].set_ylabel('2.8b Model')

fig.text(0.30, 0.95, 'Small Model Size to Large Model Size', ha='center', fontsize=20, fontweight='bold')
fig.text(0.70, 0.95, 'Large Model Size to Small Model Size', ha='center', fontsize=20, fontweight='bold')

#plt.tight_layout()
plt.savefig('combined_transition_matrix.png', bbox_inches="tight", dpi=600)
plt.show()
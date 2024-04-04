import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 加载数据
df = pd.read_csv("generate_results/memorization_evals_70m-deduped-v0_32_48_143000.csv", index_col=0)

# 计算整个数据集中每个`idx`的百分位
df['percentile_rank'] = df['idx'].rank(pct=True)

# 提取完全记忆化（score == 1）和未记忆化（score == 0）的百分位
df_full_memorization = df[df['score'] == 1]['percentile_rank']
df_unmemorized = df[df['score'] == 0]['percentile_rank']

# 创建绘图
plt.figure(figsize=(10, 6))

# 绘制百分位直方图
plt.hist(df_full_memorization, bins=30, alpha=0.75, label='Memorized Sequences', color='blue')
plt.hist(df_unmemorized, bins=30, alpha=0.75, label='Unmemorized Sequences', color='red')

# 设置标签和标题
plt.title('Percentile Distribution of Memorized and Unmemorized Sequences')
plt.xlabel('Percentile Rank')
plt.ylabel('Count')
plt.legend()

# 保存和显示图形
plt.savefig("percentile_distribution_comparison.png")
plt.show()

# 清除当前图形
plt.clf()
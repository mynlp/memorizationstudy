import pandas as pd
import matplotlib.pyplot as plt

# 加载数据
df = pd.read_csv("generate_results/memorization_evals_70m-deduped-v0_32_48_143000.csv", index_col=0)

# 绘制整体idx分布
plt.hist(df['idx'], bins=30, alpha=0.5, label='Total Distribution', color='grey')

# 提取记忆化（score为1）的idx并绘制其分布
df_full_memorization = df[df['score'] == 1]
plt.hist(df_full_memorization['idx'], bins=30, alpha=0.75, label='Memorized Sequences', color='blue')

# 提取未记忆化（score为0）的idx并绘制其分布
df_unmemorized = df[df['score'] == 0]
plt.hist(df_unmemorized['idx'], bins=30, alpha=0.75, label='Unmemorized Sequences', color='red')

# 添加图例、标题和坐标轴标签
plt.legend()
plt.title('Distribution of idx in Total, Memorized, and Unmemorized Sequences')
plt.xlabel('idx')
plt.ylabel('Count')

# 保存图像文件
plt.savefig("distribution_comparison.png")

# 显示图形
plt.show()

# 清除当前图形
plt.clf()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 步骤1: 加载数据
for model_size in ["70m", "410m", "2.8b", "12b"]:
    df = pd.read_csv(f"generate_results/memorization_evals_{model_size}-deduped-v0_32_48_143000.csv", index_col=0)
    # 提取完全记忆化（score == 1）的数据
    df_full_memorization = df[df['score'] == 1]
    df_un_memorization = df[df['score'] ==0]

    # 计算`idx`在整个数据集中的百分位
    df['percentile'] = pd.qcut(df['idx'], 50, labels=False)  # 把整个数据集的idx分成10个百分位区间

    # 计算df_full_memorization中每个idx的百分位
    df_full_memorization['percentile'] = pd.cut(df_full_memorization['idx'], bins=np.percentile(df['idx'], np.arange(0, 101, 2)), labels=False, include_lowest=True)
    df_un_memorization['percentile'] = pd.cut(df_un_memorization['idx'], bins=np.percentile(df['idx'], np.arange(0, 101, 2)), labels=False, include_lowest=True)
    # 步骤2: 计算df_full_memorization中每个百分位区间的idx数量
    counts_per_percentile = df_full_memorization['percentile'].value_counts().sort_index()
    counts_per_percentile_un = df_un_memorization['percentile'].value_counts().sort_index()
    # 输出结果
    print(counts_per_percentile)

    # 计算百分位区间的标签 (10% 到 100%)
    percentile_labels = [f"{i}-{i+2}%" for i in range(0, 100, 2)]

    # 画图
    plt.figure(figsize=(10, 6))
    counts_per_percentile.plot(kind='bar')
    plt.xticks(ticks=range(len(percentile_labels)), labels=percentile_labels, rotation=45)
    plt.xlabel('Percentile Range')
    plt.ylabel('Count of Memorized idx')
    plt.title(f'{model_size} Distribution of Memorized idx Across Percentiles')
    plt.grid(axis='y')
    # 显示图形
    plt.tight_layout()  # 调整布局以防止标签被剪切
    plt.savefig(f"{model_size} memorized_idx_distribution_percentiles.png")
    plt.show()

    plt.figure(figsize=(10, 6))
    counts_per_percentile_un.plot(kind='bar')
    plt.xticks(ticks=range(len(percentile_labels)), labels=percentile_labels, rotation=45)
    plt.xlabel('Percentile Range')
    plt.ylabel('Count of Memorized idx')
    plt.title(f'{model_size} Distribution of Memorized idx Across Percentiles')
    plt.grid(axis='y')
    # 显示图形
    plt.tight_layout()  # 调整布局以防止标签被剪切
    plt.savefig(f"{model_size} unmemorized_idx_distribution_percentiles.png")
    plt.show()

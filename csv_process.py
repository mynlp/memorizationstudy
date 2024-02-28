from utils import *
import numpy as np
import pandas as pd
for file in ["memorization_evals_70m-deduped-v0_32_48_143000.csv", "memorization_evals_70m-deduped-v0_32_64_143000.csv",
            "memorization_evals_70m-deduped-v0_32_80_143000.csv", "memorization_evals_70m-deduped-v0_32_96_143000.csv",
            "memorization_evals_70m-deduped-v0_32_128_143000.csv",
            "memorization_evals_160m-deduped-v0_32_48_143000.csv","memorization_evals_160m-deduped-v0_32_64_143000.csv",
            "memorization_evals_410m-deduped-v0_32_48_143000.csv",
                ]:
    print(file)
    generate_results = pd.read_csv("generate_results/"+file, index_col=0)
    results_list = []
    results = generate_results[generate_results['score'] == 0]
    results_list.append(results)
    #results.to_csv("/work/gk77/k77025/memorizationstudy/generate_results/memorization_evals_70m-deduped-v0_143000_0.csv")
    for i in np.arange(0, 1, 0.1):
        if i == 0.9:
            results = generate_results[generate_results['score'].between(i, i+0.1, inclusive="left")]
        elif i == 0:
            results = generate_results[generate_results['score'].between(i, i+0.1, inclusive="right")]
        else:
            results = generate_results[generate_results['score'].between(i, i+0.1, inclusive="left")]
        results_list.append(results)
        #results.to_csv(f"/work/gk77/k77025/memorizationstudy/generate_results/memorization_evals_70m-deduped-v0_143000_{i}.csv")
    results = generate_results[generate_results["score"] == 1.0]
    results_list.append(results)
    [print(len(x)) for x in results_list]


# def find_cross(df1, df2, value):
#     df1 = df1[df1['0.0'] == value]
#     df2 = df2[df2['0.0'] == value]
#     df1_index = set(df1["0"].to_list())
#     df2_index = set(df2["0"].to_list())
#     return df1_index.intersection(df2_index)
# df1 = read_csv("/work/gk77/k77025/memorizationstudy/generate_results/memorization_evals_70m-deduped-v0_32_48_143000.csv")
# df2 = read_csv("/work/gk77/k77025/memorizationstudy/generate_results/memorization_evals_70m-deduped-v0_32_64_143000.csv")
#
# cross = find_cross(df1, df2, 1)


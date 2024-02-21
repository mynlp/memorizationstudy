from utils import *
import numpy as np

generate_results = read_csv("/work/gk77/k77025/memorizationstudy/generate_results/memorization_evals_70m-deduped-v0_143000.csv")
results_list = []
results = generate_results[generate_results['0.0'] == 1.0]
results_list.append(results)
results.to_csv("/work/gk77/k77025/memorizationstudy/generate_results/memorization_evals_70m-deduped-v0_143000_1.csv")
results = generate_results[generate_results['0.0'] == 0]
results_list.append(results)
results.to_csv("/work/gk77/k77025/memorizationstudy/generate_results/memorization_evals_70m-deduped-v0_143000_0.csv")
for i in np.arange(0, 1, 0.1):
    results = generate_results[generate_results['0.0'].between(i, i+0.1, inclusive=False)]
    results_list.append(results)
    #results.to_csv(f"/work/gk77/k77025/memorizationstudy/generate_results/memorization_evals_70m-deduped-v0_143000_{i}.csv")

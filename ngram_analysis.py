import pandas
import json
from pythia.utils.mmap_dataset import MMapIndexedDataset
from tqdm import tqdm
import ijson
import pandas as pd
import random
import torch
import matplotlib.pyplot as plt

sizes = ["160m", "410m", "1b", "2.8b", "6.9b", "12b"]
context_length = ["32"]
target_length = ["48"]
num_points = 20000

memorized, half_memorized, forgotten = dict(), dict(), dict()
index_set = set()
with open("/work/gk77/share/memorizationstudy_freq/tokenid_frequency.1gram.json") as file:
    n_gram_dict = json.load(file)
def read_by_idx(mmap_ds, idx_list):
    by_index_frequency_batched = []
    for idx in idx_list:
        sent_tokens = mmap_ds[int(idx)]
        by_index_frequency = []
        for token in sent_tokens[:80]:
            by_index_frequency.append(n_gram_dict[str(token)])
        by_index_frequency_batched.append(by_index_frequency)
    return by_index_frequency_batched

mmap_ds = MMapIndexedDataset('deduped_merge/document.bin', skip_warmup=True)
for size in sizes:
    df = pd.read_csv(f"generate_results/memorization_evals_{size}-deduped-v0_32_48_143000.csv", index_col=0)
    memorized = df[df['score'] == 1]
    half_memorized = df[df['score'] == 0.5]
    forgotten = df[df['score'] == 0]
    idx_full_memorization = memorized["idx"].tolist()
    idx_not_full_memorization = half_memorized["idx"].tolist()
    idx_half_memorization = forgotten["idx"].tolist()
    memorized_index = random.sample(idx_full_memorization, num_points)
    half_memorized_index = random.sample(idx_not_full_memorization, num_points)
    forgotten_index = random.sample(idx_half_memorization, num_points)
    memorized_batched = read_by_idx(mmap_ds, memorized_index)
    half_memorized_batched = read_by_idx(mmap_ds, half_memorized_index)
    forgotten_batched = read_by_idx(mmap_ds, forgotten_index)
    averaged_memorized = torch.Tensor(memorized_batched).mean(dim=0)
    averaged_half_memorized = torch.Tensor(half_memorized_batched).mean(dim=0)
    averaged_forgotten = torch.Tensor(forgotten_batched).mean(dim=0)
    print("size: ", size)
    print(averaged_memorized[:48])
    print(averaged_half_memorized[:48])
    print(averaged_forgotten[:48])


# size_70m_memorized=[1.7204e+09, 1.7526e+09, 1.7298e+09, 1.7660e+09, 1.6903e+09, 1.7650e+09,
#         1.7450e+09, 1.7416e+09, 1.6979e+09, 1.7181e+09, 1.7516e+09, 1.7057e+09,
#         1.7135e+09, 1.6907e+09, 1.7173e+09, 1.6473e+09, 1.6855e+09, 1.7032e+09,
#         1.6853e+09, 1.7133e+09, 1.7171e+09, 1.7459e+09, 1.7043e+09, 1.7506e+09,
#         1.7020e+09, 1.7399e+09, 1.7706e+09, 1.7091e+09, 1.7829e+09, 1.6949e+09,
#         1.7531e+09, 1.7121e+09, 1.7625e+09, 1.8125e+09, 1.7703e+09, 1.7650e+09,
#         1.7413e+09, 1.7566e+09, 1.7340e+09, 1.7293e+09, 1.6619e+09, 1.7383e+09,
#         1.7366e+09, 1.6627e+09, 1.7223e+09, 1.7627e+09, 1.7878e+09, 1.8639e+09]
# sized_70m_half_memorized = [1.7475e+09, 1.7714e+09, 1.7478e+09, 1.7604e+09, 1.6938e+09, 1.7112e+09,
#         1.7009e+09, 1.7374e+09, 1.7058e+09, 1.7546e+09, 1.7377e+09, 1.7785e+09,
#         1.7215e+09, 1.7513e+09, 1.7376e+09, 1.6937e+09, 1.7390e+09, 1.7419e+09,
#         1.7382e+09, 1.7285e+09, 1.6913e+09, 1.6776e+09, 1.7403e+09, 1.7482e+09,
#         1.6917e+09, 1.7288e+09, 1.7007e+09, 1.7245e+09, 1.6583e+09, 1.6926e+09,
#         1.6555e+09, 1.6668e+09, 1.8783e+09, 1.8420e+09, 1.8426e+09, 1.8416e+09,
#         1.8718e+09, 1.8856e+09, 1.9982e+09, 2.0321e+09, 1.7232e+09, 1.7387e+09,
#         1.7074e+09, 1.7517e+09, 1.7536e+09, 1.7299e+09, 1.7330e+09, 1.7325e+09]
# size_70m_unmemorized = [1.7004e+09, 1.7455e+09, 1.7284e+09, 1.6990e+09, 1.7224e+09, 1.7156e+09,
#         1.7277e+09, 1.7037e+09, 1.7869e+09, 1.7311e+09, 1.7239e+09, 1.7370e+09,
#         1.7424e+09, 1.7278e+09, 1.7199e+09, 1.7135e+09, 1.7238e+09, 1.7451e+09,
#         1.7266e+09, 1.7170e+09, 1.7168e+09, 1.6991e+09, 1.7037e+09, 1.6841e+09,
#         1.7787e+09, 1.7349e+09, 1.7076e+09, 1.7386e+09, 1.7615e+09, 1.7480e+09,
#         1.7507e+09, 1.8471e+09, 9.4194e+08, 1.4720e+09, 1.6174e+09, 1.6410e+09,
#         1.6723e+09, 1.7107e+09, 1.7125e+09, 1.7274e+09, 1.7058e+09, 1.7015e+09,
#         1.6953e+09, 1.6792e+09, 1.6972e+09, 1.7235e+09, 1.6948e+09, 1.7020e+09]
# size_160m_memorized = [1.1907e+09, 1.9115e+09, 2.4001e+09, 1.5631e+09, 1.9958e+09, 1.8971e+09,
#         1.2424e+09, 1.1092e+09, 1.4225e+09, 1.8750e+09, 5.7415e+09, 2.1327e+09,
#         1.1123e+09, 1.1417e+09, 2.6673e+09, 1.9364e+09, 3.7032e+08, 6.4259e+08,
#         1.6471e+09, 2.5830e+09, 5.4808e+08, 9.9562e+08, 3.2626e+09, 1.1272e+09,
#         1.7927e+09, 2.6725e+09, 1.1736e+09, 3.4880e+09, 3.2586e+08, 1.5435e+09,
#         2.0703e+09, 1.0187e+09, 2.6331e+09, 2.4807e+09, 4.9746e+08, 3.2101e+09,
#         9.7018e+08, 3.0720e+09, 9.5588e+08, 9.7307e+08, 2.1002e+09, 1.7386e+09,
#         1.2712e+09, 2.7830e+09, 1.5900e+09, 1.8192e+09, 2.4229e+09, 6.4173e+08]
# size_160m_half_memorized = [1.5016e+09, 2.3836e+09, 8.3772e+08, 2.1503e+09, 1.4076e+09, 4.8565e+08,
#         9.2306e+08, 2.3759e+09, 9.7856e+08, 1.7414e+09, 7.4168e+08, 2.0384e+09,
#         7.4215e+08, 1.9174e+09, 1.6133e+09, 1.3662e+09, 2.2523e+09, 1.2728e+09,
#         1.4078e+09, 1.7520e+09, 3.9443e+09, 1.0309e+09, 1.2148e+09, 1.7074e+09,
#         1.3614e+09, 1.7712e+09, 8.3204e+08, 9.7778e+08, 8.6153e+08, 2.1183e+09,
#         2.3762e+09, 7.8617e+08, 2.1403e+09, 7.5975e+09, 2.2388e+09, 2.5412e+09,
#         1.1179e+09, 9.4249e+08, 2.9743e+09, 1.0041e+09, 2.3826e+09, 1.2437e+09,
#         9.1795e+08, 2.5773e+09, 1.6277e+09, 1.6893e+09, 1.9931e+09, 1.4799e+09]
# size_160m_unmemorized = [2.2112e+09, 1.2340e+09, 1.9110e+09, 3.0066e+09, 1.8237e+09, 2.3654e+09,
#         2.3777e+09, 8.0126e+08, 1.2615e+09, 1.2541e+09, 5.0781e+08, 2.6354e+09,
#         2.2720e+09, 1.7744e+09, 2.3389e+09, 1.7138e+09, 1.6195e+09, 2.3637e+09,
#         2.1019e+09, 1.1005e+09, 1.2944e+09, 6.0440e+08, 9.9908e+08, 1.9658e+09,
#         1.6693e+09, 9.1247e+08, 1.2257e+09, 1.0253e+09, 2.0590e+09, 1.8086e+09,
#         1.4383e+09, 2.0835e+09, 2.1358e+09, 2.8998e+09, 6.4452e+08, 1.6149e+09,
#         1.4726e+09, 9.5739e+08, 1.8834e+09, 1.7482e+09, 1.4942e+09, 2.5919e+09,
#         1.8241e+09, 6.4313e+08, 1.0021e+09, 1.8579e+09, 1.6872e+09, 1.5688e+09]
# plt.plot(range(48), size_70m_memorized, label='70m_memorized')
# plt.plot(range(48), sized_70m_half_memorized, label='70m_half_memorized')
# plt.plot(range(48), size_70m_unmemorized, label='70m_unmemorized')
# plt.plot(range(48), size_160m_memorized, label='160m_memorized')
# plt.plot(range(48), size_160m_half_memorized, label='160m_half_memorized')
# plt.plot(range(48), size_160m_unmemorized, label='160m_unmemorized')
#
# # Label the graph
# plt.title('Comparison of Memorization Rates')
# plt.xlabel('Time')
# plt.ylabel('Rate')
# plt.legend()
#
# # Display the graph
# plt.show()
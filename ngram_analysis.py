import pandas
import json
from pythia.utils.mmap_dataset import MMapIndexedDataset
from tqdm import tqdm
import ijson
import pandas as pd
import random
import torch
import matplotlib.pyplot as plt

sizes = ["70m", "160m", "410m", "1b", "2.8b", "6.9b", "12b"]
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


size_70m_memorized=[1.1120e+09, 1.0275e+09, 1.9005e+09, 1.8344e+09, 1.9544e+09, 2.1434e+09,
        1.9523e+09, 1.4976e+09, 9.5861e+08, 3.5366e+09, 6.8371e+07, 1.4888e+09,
        7.9231e+08, 2.1079e+09, 2.7326e+09, 5.9638e+08, 3.0295e+09, 2.7447e+09,
        2.5422e+09, 1.8686e+09, 2.1369e+08, 1.5382e+09, 1.5515e+09, 1.2020e+09,
        1.5263e+09, 1.4794e+09, 1.8743e+08, 2.3062e+09, 1.5923e+09, 2.6175e+09,
        4.3359e+09, 4.6660e+09, 2.4720e+09, 1.8096e+09, 2.5115e+09, 2.4681e+09,
        5.1638e+08, 1.5973e+09, 1.7925e+09, 1.8063e+09, 2.8868e+09, 1.7699e+09,
        4.8350e+08, 1.2598e+09, 4.0381e+08, 1.1378e+09, 6.3260e+08, 9.4550e+08]
sized_70m_half_memorized = [1.4538e+09, 2.7054e+09, 1.9762e+09, 2.0665e+09, 8.9114e+08, 2.3761e+09,
        2.2270e+09, 1.3922e+09, 1.9261e+09, 1.3864e+09, 1.4201e+09, 2.3350e+09,
        1.7495e+09, 5.0235e+09, 5.3640e+08, 1.6334e+09, 1.8966e+09, 7.9676e+08,
        2.2919e+09, 3.2090e+09, 1.5241e+09, 2.4531e+09, 1.7413e+09, 1.0150e+09,
        1.7930e+09, 2.0105e+09, 1.7792e+09, 2.4640e+09, 1.5988e+09, 2.3922e+09,
        7.0133e+08, 7.5204e+08, 1.0961e+09, 2.0513e+09, 9.0935e+08, 3.9910e+08,
        1.5953e+09, 1.6796e+09, 8.9171e+08, 1.3838e+09, 1.9984e+09, 5.6416e+08,
        1.8483e+09, 1.6069e+09, 1.4536e+09, 1.4449e+09, 1.9938e+09, 7.9358e+08]
size_70m_unmemorized = [9.6449e+08, 1.7586e+09, 1.6704e+09, 6.6660e+08, 1.7808e+09, 2.0748e+09,
        2.3167e+09, 1.8434e+09, 1.9175e+09, 1.4656e+09, 2.0179e+09, 1.1240e+09,
        1.6742e+09, 2.0091e+09, 1.4871e+09, 4.0854e+08, 2.1787e+09, 1.4189e+09,
        2.8242e+09, 2.8142e+09, 1.6907e+09, 1.7181e+09, 1.6563e+09, 2.0594e+09,
        1.2481e+09, 9.0999e+08, 1.6332e+09, 2.4825e+09, 2.7269e+09, 1.7018e+09,
        1.2513e+09, 1.0865e+09, 1.4696e+09, 2.5888e+09, 2.1071e+09, 1.0246e+09,
        2.3877e+09, 2.7999e+09, 1.8891e+09, 1.3424e+09, 1.1528e+09, 1.1543e+09,
        1.0804e+09, 3.1868e+09, 1.8825e+09, 1.7334e+09, 1.8313e+09, 1.4326e+09]
size_160m_memorized = [1.1907e+09, 1.9115e+09, 2.4001e+09, 1.5631e+09, 1.9958e+09, 1.8971e+09,
        1.2424e+09, 1.1092e+09, 1.4225e+09, 1.8750e+09, 5.7415e+09, 2.1327e+09,
        1.1123e+09, 1.1417e+09, 2.6673e+09, 1.9364e+09, 3.7032e+08, 6.4259e+08,
        1.6471e+09, 2.5830e+09, 5.4808e+08, 9.9562e+08, 3.2626e+09, 1.1272e+09,
        1.7927e+09, 2.6725e+09, 1.1736e+09, 3.4880e+09, 3.2586e+08, 1.5435e+09,
        2.0703e+09, 1.0187e+09, 2.6331e+09, 2.4807e+09, 4.9746e+08, 3.2101e+09,
        9.7018e+08, 3.0720e+09, 9.5588e+08, 9.7307e+08, 2.1002e+09, 1.7386e+09,
        1.2712e+09, 2.7830e+09, 1.5900e+09, 1.8192e+09, 2.4229e+09, 6.4173e+08]
size_160m_half_memorized = [1.5016e+09, 2.3836e+09, 8.3772e+08, 2.1503e+09, 1.4076e+09, 4.8565e+08,
        9.2306e+08, 2.3759e+09, 9.7856e+08, 1.7414e+09, 7.4168e+08, 2.0384e+09,
        7.4215e+08, 1.9174e+09, 1.6133e+09, 1.3662e+09, 2.2523e+09, 1.2728e+09,
        1.4078e+09, 1.7520e+09, 3.9443e+09, 1.0309e+09, 1.2148e+09, 1.7074e+09,
        1.3614e+09, 1.7712e+09, 8.3204e+08, 9.7778e+08, 8.6153e+08, 2.1183e+09,
        2.3762e+09, 7.8617e+08, 2.1403e+09, 7.5975e+09, 2.2388e+09, 2.5412e+09,
        1.1179e+09, 9.4249e+08, 2.9743e+09, 1.0041e+09, 2.3826e+09, 1.2437e+09,
        9.1795e+08, 2.5773e+09, 1.6277e+09, 1.6893e+09, 1.9931e+09, 1.4799e+09]
size_160m_unmemorized = [2.2112e+09, 1.2340e+09, 1.9110e+09, 3.0066e+09, 1.8237e+09, 2.3654e+09,
        2.3777e+09, 8.0126e+08, 1.2615e+09, 1.2541e+09, 5.0781e+08, 2.6354e+09,
        2.2720e+09, 1.7744e+09, 2.3389e+09, 1.7138e+09, 1.6195e+09, 2.3637e+09,
        2.1019e+09, 1.1005e+09, 1.2944e+09, 6.0440e+08, 9.9908e+08, 1.9658e+09,
        1.6693e+09, 9.1247e+08, 1.2257e+09, 1.0253e+09, 2.0590e+09, 1.8086e+09,
        1.4383e+09, 2.0835e+09, 2.1358e+09, 2.8998e+09, 6.4452e+08, 1.6149e+09,
        1.4726e+09, 9.5739e+08, 1.8834e+09, 1.7482e+09, 1.4942e+09, 2.5919e+09,
        1.8241e+09, 6.4313e+08, 1.0021e+09, 1.8579e+09, 1.6872e+09, 1.5688e+09]
plt.plot(range(48), size_70m_memorized, label='70m_memorized')
plt.plot(range(48), sized_70m_half_memorized, label='70m_half_memorized')
plt.plot(range(48), size_70m_unmemorized, label='70m_unmemorized')
plt.plot(range(48), size_160m_memorized, label='160m_memorized')
plt.plot(range(48), size_160m_half_memorized, label='160m_half_memorized')
plt.plot(range(48), size_160m_unmemorized, label='160m_unmemorized')

# Label the graph
plt.title('Comparison of Memorization Rates')
plt.xlabel('Time')
plt.ylabel('Rate')
plt.legend()

# Display the graph
plt.show()
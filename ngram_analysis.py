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
    #memorized = df[df['score'] == 1]
    #half_memorized = df[df['score'] == 0.5]
    #forgotten = df[df['score'] == 0]
    quarter_memorized = df[df['score'] == 0.25]
    #idx_full_memorization = memorized["idx"].tolist()
    #idx_not_full_memorization = half_memorized["idx"].tolist()
    #idx_half_memorization = forgotten["idx"].tolist()
    idx_quarter_memorization = quarter_memorized["idx"].tolist()
    #memorized_index = random.sample(idx_full_memorization, num_points)
    #half_memorized_index = random.sample(idx_not_full_memorization, num_points)
    #forgotten_index = random.sample(idx_half_memorization, num_points)
    quarter_memorized_index = random.sample(idx_quarter_memorization, num_points)
    #memorized_batched = read_by_idx(mmap_ds, memorized_index)
    #half_memorized_batched = read_by_idx(mmap_ds, half_memorized_index)
    #forgotten_batched = read_by_idx(mmap_ds, forgotten_index)
    quarter_memorized_batched = read_by_idx(mmap_ds, quarter_memorized_index)
    #averaged_memorized = torch.Tensor(memorized_batched).mean(dim=0)
    #veraged_half_memorized = torch.Tensor(half_memorized_batched).mean(dim=0)
    #averaged_forgotten = torch.Tensor(forgotten_batched).mean(dim=0)
    averaged_quarter_memorized = torch.Tensor(quarter_memorized_batched).mean(dim=0)
    print("size: ", size)
    #print(averaged_memorized[:48])
    #print(averaged_half_memorized[:48])
    #print(averaged_forgotten[:48])
    print(averaged_quarter_memorized[:48])

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
# size_160m_memorized = [1.7581e+09, 1.7453e+09, 1.7578e+09, 1.7682e+09, 1.7401e+09, 1.7368e+09,
#         1.6692e+09, 1.6842e+09, 1.7073e+09, 1.6873e+09, 1.6361e+09, 1.6988e+09,
#         1.6551e+09, 1.6832e+09, 1.6740e+09, 1.7045e+09, 1.6855e+09, 1.6499e+09,
#         1.6668e+09, 1.7148e+09, 1.7334e+09, 1.6861e+09, 1.6983e+09, 1.7350e+09,
#         1.6911e+09, 1.6592e+09, 1.7162e+09, 1.7568e+09, 1.7729e+09, 1.7106e+09,
#         1.6952e+09, 1.7416e+09, 1.8137e+09, 1.7766e+09, 1.7587e+09, 1.7389e+09,
#         1.7813e+09, 1.6816e+09, 1.7227e+09, 1.7328e+09, 1.6455e+09, 1.6610e+09,
#         1.6358e+09, 1.7301e+09, 1.7271e+09, 1.7473e+09, 1.7761e+09, 1.8069e+09]
# size_160m_half_memorized = [1.7802e+09, 1.7359e+09, 1.7454e+09, 1.7121e+09, 1.7207e+09, 1.7464e+09,
#         1.7171e+09, 1.7439e+09, 1.7213e+09, 1.7561e+09, 1.7266e+09, 1.7386e+09,
#         1.6963e+09, 1.7324e+09, 1.7319e+09, 1.7257e+09, 1.7499e+09, 1.6942e+09,
#         1.7033e+09, 1.7256e+09, 1.7614e+09, 1.7386e+09, 1.7136e+09, 1.6661e+09,
#         1.7353e+09, 1.6566e+09, 1.7152e+09, 1.6454e+09, 1.7125e+09, 1.6675e+09,
#         1.6907e+09, 1.5411e+09, 1.8713e+09, 1.8547e+09, 1.9008e+09, 1.8755e+09,
#         1.9015e+09, 1.9218e+09, 2.1013e+09, 2.1584e+09, 1.6906e+09, 1.7316e+09,
#         1.7952e+09, 1.7080e+09, 1.7420e+09, 1.7208e+09, 1.7172e+09, 1.7156e+09]
# size_160m_unmemorized = [1.7139e+09, 1.7571e+09, 1.7740e+09, 1.7161e+09, 1.7512e+09, 1.7409e+09,
#         1.7519e+09, 1.7548e+09, 1.7494e+09, 1.7669e+09, 1.6993e+09, 1.6981e+09,
#         1.7514e+09, 1.6992e+09, 1.7798e+09, 1.7167e+09, 1.7292e+09, 1.7378e+09,
#         1.7370e+09, 1.7452e+09, 1.7480e+09, 1.7426e+09, 1.7166e+09, 1.7599e+09,
#         1.7309e+09, 1.7306e+09, 1.7135e+09, 1.7444e+09, 1.7604e+09, 1.7284e+09,
#         1.7686e+09, 1.8918e+09, 9.6200e+08, 1.4632e+09, 1.6455e+09, 1.6643e+09,
#         1.6458e+09, 1.7043e+09, 1.7083e+09, 1.7151e+09, 1.7159e+09, 1.7356e+09,
#         1.6756e+09, 1.6919e+09, 1.6968e+09, 1.7123e+09, 1.6960e+09, 1.6792e+09]
# size_410m_memorized = [1.7390e+09, 1.7312e+09, 1.7107e+09, 1.7662e+09, 1.7262e+09, 1.7208e+09,
#         1.6971e+09, 1.7676e+09, 1.6954e+09, 1.7309e+09, 1.7027e+09, 1.6938e+09,
#         1.7326e+09, 1.6560e+09, 1.7343e+09, 1.6603e+09, 1.6978e+09, 1.6706e+09,
#         1.6912e+09, 1.6504e+09, 1.7006e+09, 1.6979e+09, 1.6582e+09, 1.6749e+09,
#         1.6974e+09, 1.7250e+09, 1.7233e+09, 1.6808e+09, 1.7812e+09, 1.6719e+09,
#         1.6891e+09, 1.6735e+09, 1.7723e+09, 1.8046e+09, 1.7620e+09, 1.7359e+09,
#         1.7220e+09, 1.7168e+09, 1.7016e+09, 1.6847e+09, 1.7043e+09, 1.6673e+09,
#         1.7164e+09, 1.7109e+09, 1.7087e+09, 1.7419e+09, 1.8182e+09, 1.8000e+09]
# size_410m_half_memorized = [1.7529e+09, 1.7076e+09, 1.7585e+09, 1.7437e+09, 1.7265e+09, 1.7623e+09,
#         1.6736e+09, 1.7102e+09, 1.7282e+09, 1.7623e+09, 1.7610e+09, 1.6918e+09,
#         1.6784e+09, 1.7080e+09, 1.7355e+09, 1.7749e+09, 1.7408e+09, 1.7421e+09,
#         1.7407e+09, 1.7291e+09, 1.7148e+09, 1.6994e+09, 1.7162e+09, 1.6644e+09,
#         1.7165e+09, 1.7266e+09, 1.6542e+09, 1.7544e+09, 1.7441e+09, 1.6867e+09,
#         1.6582e+09, 1.6371e+09, 1.8003e+09, 1.8362e+09, 1.8435e+09, 1.8837e+09,
#         1.9326e+09, 1.9979e+09, 2.0817e+09, 2.1681e+09, 1.7153e+09, 1.6997e+09,
#         1.7197e+09, 1.7205e+09, 1.7648e+09, 1.7152e+09, 1.7415e+09, 1.7485e+09]
# size_410m_unmemorized = [1.7731e+09, 1.7119e+09, 1.7124e+09, 1.6960e+09, 1.7358e+09, 1.7661e+09,
#         1.7333e+09, 1.7457e+09, 1.7688e+09, 1.7591e+09, 1.7401e+09, 1.7324e+09,
#         1.7226e+09, 1.7306e+09, 1.6895e+09, 1.7556e+09, 1.7515e+09, 1.7255e+09,
#         1.7320e+09, 1.7181e+09, 1.7156e+09, 1.8104e+09, 1.7388e+09, 1.7677e+09,
#         1.7490e+09, 1.7463e+09, 1.7586e+09, 1.7629e+09, 1.7377e+09, 1.8117e+09,
#         1.7426e+09, 1.9118e+09, 9.8174e+08, 1.4030e+09, 1.5810e+09, 1.6338e+09,
#         1.6753e+09, 1.7050e+09, 1.7095e+09, 1.6390e+09, 1.7085e+09, 1.7289e+09,
#         1.6834e+09, 1.7107e+09, 1.7146e+09, 1.7228e+09, 1.6977e+09, 1.6989e+09]
# size_1b_memorized = [1.7328e+09, 1.7596e+09, 1.7529e+09, 1.7307e+09, 1.7247e+09, 1.7499e+09,
#         1.7506e+09, 1.7410e+09, 1.7103e+09, 1.7470e+09, 1.7629e+09, 1.7459e+09,
#         1.6905e+09, 1.7163e+09, 1.6949e+09, 1.6677e+09, 1.6991e+09, 1.7033e+09,
#         1.7092e+09, 1.7305e+09, 1.7251e+09, 1.7068e+09, 1.6837e+09, 1.7158e+09,
#         1.7169e+09, 1.7020e+09, 1.7236e+09, 1.7082e+09, 1.7415e+09, 1.6774e+09,
#         1.6722e+09, 1.6537e+09, 1.8078e+09, 1.7602e+09, 1.8304e+09, 1.7181e+09,
#         1.7213e+09, 1.7329e+09, 1.7053e+09, 1.6905e+09, 1.7119e+09, 1.7315e+09,
#         1.7516e+09, 1.7211e+09, 1.6991e+09, 1.7963e+09, 1.7913e+09, 1.8493e+09]
# size_1b_half_memorized = [1.7001e+09, 1.7178e+09, 1.7318e+09, 1.7452e+09, 1.6969e+09, 1.7229e+09,
#         1.7535e+09, 1.7413e+09, 1.7232e+09, 1.7224e+09, 1.7155e+09, 1.7535e+09,
#         1.6990e+09, 1.7004e+09, 1.6827e+09, 1.7323e+09, 1.7777e+09, 1.7712e+09,
#         1.7119e+09, 1.7196e+09, 1.7612e+09, 1.7371e+09, 1.6680e+09, 1.7291e+09,
#         1.7223e+09, 1.7002e+09, 1.6942e+09, 1.7078e+09, 1.6608e+09, 1.6557e+09,
#         1.6473e+09, 1.5640e+09, 1.8347e+09, 1.8292e+09, 1.8626e+09, 1.8943e+09,
#         1.9313e+09, 1.9158e+09, 2.1159e+09, 2.1892e+09, 1.6925e+09, 1.7099e+09,
#         1.7213e+09, 1.7136e+09, 1.6940e+09, 1.7217e+09, 1.7051e+09, 1.7882e+09]
# size_1b_unmemorized = [1.7116e+09, 1.7162e+09, 1.7777e+09, 1.7075e+09, 1.7517e+09, 1.7614e+09,
#         1.7520e+09, 1.7423e+09, 1.7535e+09, 1.7513e+09, 1.7281e+09, 1.7756e+09,
#         1.7558e+09, 1.7334e+09, 1.7369e+09, 1.7571e+09, 1.7904e+09, 1.7390e+09,
#         1.7310e+09, 1.7540e+09, 1.7513e+09, 1.7890e+09, 1.7319e+09, 1.7591e+09,
#         1.7546e+09, 1.7534e+09, 1.7830e+09, 1.7335e+09, 1.7316e+09, 1.7996e+09,
#         1.7345e+09, 1.9618e+09, 9.8549e+08, 1.4666e+09, 1.5785e+09, 1.6485e+09,
#         1.6442e+09, 1.6740e+09, 1.6955e+09, 1.6898e+09, 1.7479e+09, 1.7170e+09,
#         1.7631e+09, 1.6798e+09, 1.7182e+09, 1.7069e+09, 1.7225e+09, 1.7168e+09]
# size_2_8b_memorized = [1.8086e+09, 1.7658e+09, 1.7372e+09, 1.7486e+09, 1.7520e+09, 1.7526e+09,
#         1.7726e+09, 1.7103e+09, 1.7715e+09, 1.7235e+09, 1.7116e+09, 1.7054e+09,
#         1.7214e+09, 1.7272e+09, 1.7040e+09, 1.7159e+09, 1.6934e+09, 1.7213e+09,
#         1.6803e+09, 1.6903e+09, 1.7148e+09, 1.7392e+09, 1.7172e+09, 1.7536e+09,
#         1.7065e+09, 1.7689e+09, 1.7521e+09, 1.6999e+09, 1.7538e+09, 1.6992e+09,
#         1.7133e+09, 1.6696e+09, 1.8010e+09, 1.7504e+09, 1.7112e+09, 1.7603e+09,
#         1.7387e+09, 1.7270e+09, 1.7099e+09, 1.7489e+09, 1.7171e+09, 1.7299e+09,
#         1.7091e+09, 1.7298e+09, 1.7098e+09, 1.7325e+09, 1.7913e+09, 1.7937e+09]
# size_2_8b_half_memorized = [1.7295e+09, 1.7566e+09, 1.7050e+09, 1.7266e+09, 1.7482e+09, 1.6807e+09,
#         1.7745e+09, 1.7662e+09, 1.7209e+09, 1.7674e+09, 1.7246e+09, 1.7041e+09,
#         1.7141e+09, 1.6988e+09, 1.6952e+09, 1.7133e+09, 1.7473e+09, 1.7212e+09,
#         1.7057e+09, 1.7257e+09, 1.6959e+09, 1.6914e+09, 1.6981e+09, 1.6771e+09,
#         1.7331e+09, 1.6905e+09, 1.7037e+09, 1.7249e+09, 1.7395e+09, 1.6571e+09,
#         1.6832e+09, 1.5134e+09, 1.8125e+09, 1.8694e+09, 1.7784e+09, 1.8668e+09,
#         1.9261e+09, 1.9810e+09, 2.1567e+09, 2.2463e+09, 1.6916e+09, 1.7066e+09,
#         1.7433e+09, 1.7334e+09, 1.7397e+09, 1.7865e+09, 1.7506e+09, 1.7101e+09]
# size_2_8b_unmemorized = [1.7578e+09, 1.7545e+09, 1.7198e+09, 1.6863e+09, 1.7673e+09, 1.7582e+09,
#         1.7807e+09, 1.7087e+09, 1.7524e+09, 1.7594e+09, 1.7602e+09, 1.7436e+09,
#         1.7683e+09, 1.6981e+09, 1.7405e+09, 1.7523e+09, 1.7503e+09, 1.7542e+09,
#         1.7237e+09, 1.7167e+09, 1.7769e+09, 1.7473e+09, 1.7638e+09, 1.7337e+09,
#         1.7493e+09, 1.7732e+09, 1.7826e+09, 1.7324e+09, 1.7648e+09, 1.7896e+09,
#         1.8631e+09, 1.9754e+09, 1.0171e+09, 1.4158e+09, 1.5606e+09, 1.6252e+09,
#         1.6421e+09, 1.7273e+09, 1.6746e+09, 1.7352e+09, 1.7096e+09, 1.7138e+09,
#         1.6935e+09, 1.6993e+09, 1.7327e+09, 1.6830e+09, 1.7406e+09, 1.7304e+09]
# size_6_9b_memorized =[1.7634e+09, 1.7120e+09, 1.7580e+09, 1.7003e+09, 1.7664e+09, 1.7385e+09,
#         1.7580e+09, 1.6984e+09, 1.6791e+09, 1.7764e+09, 1.6723e+09, 1.7193e+09,
#         1.7397e+09, 1.7310e+09, 1.7257e+09, 1.7527e+09, 1.7199e+09, 1.7183e+09,
#         1.6893e+09, 1.7356e+09, 1.7033e+09, 1.7265e+09, 1.6882e+09, 1.7286e+09,
#         1.7584e+09, 1.7241e+09, 1.7177e+09, 1.7302e+09, 1.6863e+09, 1.6990e+09,
#         1.7119e+09, 1.6492e+09, 1.7392e+09, 1.7485e+09, 1.7693e+09, 1.7328e+09,
#         1.7286e+09, 1.7129e+09, 1.6919e+09, 1.7064e+09, 1.6984e+09, 1.7060e+09,
#         1.7068e+09, 1.7301e+09, 1.7524e+09, 1.7220e+09, 1.7964e+09, 1.8498e+09]
# size_6_9b_half_memorized = [1.7337e+09, 1.7314e+09, 1.6872e+09, 1.7535e+09, 1.6973e+09, 1.7157e+09,
#         1.7358e+09, 1.6884e+09, 1.6761e+09, 1.7728e+09, 1.7542e+09, 1.7443e+09,
#         1.7189e+09, 1.7546e+09, 1.7016e+09, 1.7753e+09, 1.7493e+09, 1.7413e+09,
#         1.7046e+09, 1.6796e+09, 1.7028e+09, 1.6809e+09, 1.7188e+09, 1.7345e+09,
#         1.6996e+09, 1.6979e+09, 1.6844e+09, 1.7218e+09, 1.6783e+09, 1.6767e+09,
#         1.6396e+09, 1.5794e+09, 1.7195e+09, 1.8345e+09, 1.8121e+09, 1.8434e+09,
#         1.8723e+09, 2.0771e+09, 2.2091e+09, 2.2762e+09, 1.6919e+09, 1.6687e+09,
#         1.7009e+09, 1.7276e+09, 1.6865e+09, 1.6972e+09, 1.7210e+09, 1.7315e+09]
# size_6_9b_unmemorized =[1.7173e+09, 1.7240e+09, 1.7303e+09, 1.7659e+09, 1.7333e+09, 1.7179e+09,
#         1.7425e+09, 1.7265e+09, 1.7756e+09, 1.7228e+09, 1.8079e+09, 1.8126e+09,
#         1.7547e+09, 1.7458e+09, 1.6990e+09, 1.7050e+09, 1.7813e+09, 1.7528e+09,
#         1.7645e+09, 1.7785e+09, 1.7613e+09, 1.7302e+09, 1.7388e+09, 1.7453e+09,
#         1.7617e+09, 1.7649e+09, 1.7371e+09, 1.7416e+09, 1.7529e+09, 1.7907e+09,
#         1.8252e+09, 1.9872e+09, 1.0243e+09, 1.4188e+09, 1.5760e+09, 1.6549e+09,
#         1.6831e+09, 1.6997e+09, 1.6973e+09, 1.7030e+09, 1.7492e+09, 1.6882e+09,
#         1.6807e+09, 1.7184e+09, 1.7191e+09, 1.7537e+09, 1.7279e+09, 1.7163e+09]
# size_12b_memorized = [1.7949e+09, 1.7425e+09, 1.8040e+09, 1.8185e+09, 1.7088e+09, 1.7104e+09,
#         1.7429e+09, 1.7314e+09, 1.7090e+09, 1.7169e+09, 1.7334e+09, 1.7295e+09,
#         1.7118e+09, 1.6905e+09, 1.6997e+09, 1.7207e+09, 1.6998e+09, 1.7274e+09,
#         1.6853e+09, 1.6989e+09, 1.7504e+09, 1.7226e+09, 1.7194e+09, 1.7265e+09,
#         1.7221e+09, 1.7085e+09, 1.7104e+09, 1.6975e+09, 1.6792e+09, 1.6997e+09,
#         1.7105e+09, 1.6834e+09, 1.7920e+09, 1.7340e+09, 1.7358e+09, 1.7358e+09,
#         1.7066e+09, 1.7873e+09, 1.7614e+09, 1.6602e+09, 1.7321e+09, 1.7292e+09,
#         1.6964e+09, 1.7373e+09, 1.6807e+09, 1.7400e+09, 1.7739e+09, 1.8532e+09]
# size_12b_half_memorized = [1.7155e+09, 1.7296e+09, 1.7741e+09, 1.7638e+09, 1.7235e+09, 1.7140e+09,
#         1.7710e+09, 1.7306e+09, 1.7382e+09, 1.7088e+09, 1.7575e+09, 1.7659e+09,
#         1.7654e+09, 1.7192e+09, 1.7308e+09, 1.7439e+09, 1.7028e+09, 1.7830e+09,
#         1.6864e+09, 1.7470e+09, 1.7334e+09, 1.6977e+09, 1.7226e+09, 1.7312e+09,
#         1.6786e+09, 1.7328e+09, 1.7335e+09, 1.6729e+09, 1.7010e+09, 1.7075e+09,
#         1.6282e+09, 1.5541e+09, 1.7917e+09, 1.7494e+09, 1.8326e+09, 1.8686e+09,
#         1.8932e+09, 2.0381e+09, 2.2545e+09, 2.3690e+09, 1.7191e+09, 1.7060e+09,
#         1.7258e+09, 1.6872e+09, 1.7210e+09, 1.7337e+09, 1.7374e+09, 1.7125e+09]
# size_12b_unmemorized = [1.7532e+09, 1.7599e+09, 1.7556e+09, 1.7798e+09, 1.7592e+09, 1.7399e+09,
#         1.7134e+09, 1.7614e+09, 1.7405e+09, 1.7648e+09, 1.7538e+09, 1.7587e+09,
#         1.7718e+09, 1.7451e+09, 1.7285e+09, 1.7462e+09, 1.7526e+09, 1.7802e+09,
#         1.7200e+09, 1.7395e+09, 1.7523e+09, 1.7279e+09, 1.7579e+09, 1.7645e+09,
#         1.7604e+09, 1.7396e+09, 1.7375e+09, 1.7535e+09, 1.7672e+09, 1.8106e+09,
#         1.8184e+09, 2.0058e+09, 9.8933e+08, 1.4179e+09, 1.5926e+09, 1.6359e+09,
#         1.6431e+09, 1.6992e+09, 1.6792e+09, 1.6948e+09, 1.7172e+09, 1.7213e+09,
#         1.6997e+09, 1.7020e+09, 1.7159e+09, 1.6961e+09, 1.7247e+09, 1.6756e+09]
#
#
#
# import numpy as np
# plt.figure(figsize=(12, 12))
# plt.rcParams['axes.labelsize'] = 14
# plt.rcParams['xtick.labelsize'] = 12
# plt.rcParams['ytick.labelsize'] = 12
# plt.rcParams['legend.fontsize'] = 10
# plt.rcParams['figure.titlesize'] = 16
# begin = 28
# end = 48
# memorized_data = np.array(
#     [size_410m_memorized[begin:end], size_2_8b_memorized[begin:end], size_12b_memorized[begin:end]])
# half_memorized_data = np.array(
#     [size_410m_half_memorized[begin:end], size_2_8b_half_memorized[begin:end], size_12b_half_memorized[begin:end]])
# unmemorized_data = np.array(
#     [size_410m_unmemorized[begin:end], size_2_8b_unmemorized[begin:end], size_12b_unmemorized[begin:end]])
# avg_memorized = np.average(memorized_data, axis=0)
# min_memorized = np.min(memorized_data, axis=0)
# max_memorized = np.max(memorized_data, axis=0)
# min_half_memorized = np.min(half_memorized_data, axis=0)
# max_half_memorized = np.max(half_memorized_data, axis=0)
# min_unmemorized = np.min(unmemorized_data, axis=0)
# max_unmemorized = np.max(unmemorized_data, axis=0)
# # plt.plot(range(begin,end), size_70m_memorized[begin:end], label='70m_memorized')
# # plt.plot(range(begin,end), sized_70m_half_memorized[begin:end], label='70m_half_memorized')
# # plt.plot(range(begin,end), size_70m_unmemorized[begin:end], label='70m_unmemorized')
# # plt.plot(range(end), size_160m_memorized, label='160m_memorized')
# # plt.plot(range(end), size_160m_half_memorized, label='160m_half_memorized')
# # plt.plot(range(end), size_160m_unmemorized, label='160m_unmemorized')
# plt.plot(range(begin,end), size_410m_memorized[begin:end], label='410m_memorized')
# plt.plot(range(begin,end), size_410m_half_memorized[begin:end], label='410m_half_memorized')
# plt.plot(range(begin,end), size_410m_unmemorized[begin:end], label='410m_unmemorized')
# # plt.plot(range(begin,end), size_1b_memorized[begin:end], label='1b_memorized')
# # plt.plot(range(begin,end), size_1b_half_memorized[begin:end], label='1b_half_memorized')
# # plt.plot(range(begin,end), size_1b_unmemorized[begin:end], label='1b_unmemorized')
# plt.plot(range(begin,end), size_2_8b_memorized[begin:end], label='2.8b_memorized')
# plt.plot(range(begin,end), size_2_8b_half_memorized[begin:end], label='2.8b_half_memorized')
# plt.plot(range(begin,end), size_2_8b_unmemorized[begin:end], label='2.8b_unmemorized')
# # plt.plot(range(end), size_6_9b_memorized, label='6.9b_memorized')
# # plt.plot(range(end), size_6_9b_half_memorized, label='6.9b_half_memorized')
# # plt.plot(range(end), size_6_9b_unmemorized, label='6.9b_unmemorized')
# plt.plot(range(begin,end), size_12b_memorized[begin:end], label='12b_memorized')
# plt.plot(range(begin,end), size_12b_half_memorized[begin:end], label='12b_half_memorized')
# plt.plot(range(begin,end), size_12b_unmemorized[begin:end], label='12b_unmemorized')
#
# plt.fill_between(range(begin, end), min_memorized, max_memorized, color="blue", alpha=0.35, label='memorized')
# plt.fill_between(range(begin, end), min_half_memorized, max_half_memorized, color="orange", alpha=0.35, label='half memorized')
# plt.fill_between(range(begin, end), min_unmemorized, max_unmemorized, color="green", alpha=0.35, label='unmemorized')
#
# # Add vertical line at context points
# plt.axvline(x=31, color='red', linestyle='--')
# plt.text(31-1.8, 2*1e9, 'Context\nEnd Point', rotation=0, size=12)
# plt.axvline(x=32, color='blue', linestyle='--')
# plt.text(32+0.3, 2*1e9, 'Decoding\nStart Point', rotation=0, size=12)
#
# # calculate min, max and plot shaded area for other categories as before
#
# # Label the graph
# plt.title('One-gram Frequency Memorization Analysis at Each Sentence Index', fontsize=16)
# plt.xlabel('Index of the Sentence', fontsize=16)
# plt.ylabel('Frequency', fontsize=16)
# plt.legend(fontsize=12)
# plt.grid(True)
# #plt.yscale('log')
# # plt.title('Comparison of Memorization Rates', fontsize=16)
# # plt.xlabel('Time', fontsize=14)
# # plt.ylabel('Rate', fontsize=14)
# # plt.legend(fontsize=10)
# # Display the graph
# plt.savefig("one-gram.png", dpi=600)
# plt.show()
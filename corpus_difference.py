from pythia.utils.mmap_dataset import MMapIndexedDataset
from utils import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
import pandas as pd
import json

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

random.seed(42)
small_model_size = "70m"
large_model_size = "410m"
context = 32
continuation = 16
prefix = 'deduped_merge/document.bin'
print(prefix)
buff_size = 2049*1024*2
print("Building dataset")
#mmap_ds = MMapIndexedDataset(prefix, skip_warmup=True)

df_small = pd.read_csv(f"generate_results/memorization_evals_{small_model_size}-deduped-v0_{context}_{context+continuation}_143000.csv", index_col=0)
df_large = pd.read_csv(f"generate_results/memorization_evals_{large_model_size}-deduped-v0_{context}_{context+continuation}_143000.csv", index_col=0)
df_small_memorized = df_small[df_small["score"] == 1]
df_large_memorized = df_large[df_large["score"] == 1]

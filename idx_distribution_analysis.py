import pandas as pd
from transformers import GPTNeoXForCausalLM, AutoTokenizer,  AutoModelForCausalLM,  LogitsProcessorList, MinLengthLogitsProcessor, StoppingCriteriaList,  MaxLengthCriteria
from pythia.utils.mmap_dataset import MMapIndexedDataset
from transformers import GPTNeoXForCausalLM, AutoTokenizer
import torch
from utils import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random

df = pd.read_csv("generate_results/memorization_evals_70m-deduped-v0_32_48_143000.csv", index_col=0)
df_full_memorization = df[df['score'] == 1]
df_full_memorization["idx"].plot(kind="hist", bins=30, title="Distribution of memorized sequences")
df_full_memorization = df[df['score'] == 0]
df_full_memorization["idx"].plot(kind="hist", bins=30, title="Distribution of unmemorized sequences")

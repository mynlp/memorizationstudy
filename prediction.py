import numpy as np
import torch
import pyro
from pyro.nn import PyroModule, PyroSample
import pyro.distributions as dist
import torch.nn as nn
import random
from datasets import Dataset

random.seed(42)
model_size = "70m"
model_name = f"EleutherAI/pythia-{model_size}-deduped-v0"
CHECKPOINT = 143000
continuation = 16
dataset = {str(i): {"label":i/continuation,"token":[]} for i in range(continuation)}

for i in range(continuation):
    dataset[str(i)]["token"]=torch.load(f"cross_remembered/context_tokens_{continuation}_{i}_{model_size}.pt")
    dataset[str(i)]["label"]=i/continuation
dataset = Dataset.from_dict(dataset)

import pdb
from matplotlib import pyplot as plt
import numpy as np
import torch
import random
from datasets import Dataset
from transformers import GPTNeoXForCausalLM
from tqdm import tqdm
from models import *
from torch.utils.data import DataLoader
import argparse
import pdb
import os
from datasets import DatasetDict


def format_example(example):
    tokens, labels, embeddings, prediction, entropy = example['token'], example['label'], example["embedding"], example["prediction"], example["entropy"]
    return {'input_ids': tokens, 'labels': labels, 'embedding': embeddings, 'prediction': prediction, 'entropy': entropy}


args = argparse.ArgumentParser()
args.add_argument("--model_size", type=str, default="70m")
args.add_argument("--context_size", type=int, default=32)
args.add_argument("--continuation_size", type=int, default=16)
args.add_argument("--checkpoint", type=int, default=143000)
args.add_argument("--seed", type=int, default=42)
args.add_argument("--epoch", type=int, default=30)
args.add_argument("--hidden_size", type=int, default=512)
args.add_argument("--load_cache", type=bool, default=False)
args.add_argument("--model_type", type=str, default="transformer")
args.add_argument("--batch_size", type=int, default=128)
args.add_argument("--lr", type=float, default=1e-4)
args = args.parse_args()
print(args)
embedding_size_dict = {"70m": 512, "160m": 768, "410m": 1024, "1b": 2048, "2.8b": 2560, "6.9b": 4096, "12b": 5120}
embedding_size = embedding_size_dict[args.model_size]
random.seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
model_name = f"EleutherAI/pythia-{args.model_size}-deduped-v0"
if num_gpus > 1:
    print("Number of available GPUs: ", num_gpus)
else:
    print("Number of available GPU: ", num_gpus)
if args.load_cache == False:
    dataset = {"token": torch.empty((0,)), "label": torch.empty((0,)), "embedding": torch.empty((0,)),
               "prediction": torch.empty((0,)), "entropy": torch.empty((0,))}
    for i in tqdm(range(args.continuation_size)):
        local_data = torch.load(
            f"cross_remembered/context_tokens_{args.context_size}_{args.continuation_size}_{i}_{args.model_size}.pt",
            map_location="cpu")
        local_embedding = torch.load(
            f"cross_remembered/embeddings_{args.context_size}_{args.continuation_size}_{i}_{args.model_size}.pt",
            map_location="cpu")
        local_entropy = torch.load(
            f"cross_remembered/entropy_{args.context_size}_{args.continuation_size}_{i}_{args.model_size}.pt",
            map_location="cpu")
        local_memorized = torch.load(
            f"cross_remembered/memorized_idx_{args.context_size}_{args.continuation_size}_{i}_{args.model_size}.pt",
            map_location="cpu")
        dataset["token"] = torch.cat((dataset["token"], local_data))
        dataset["label"] = torch.cat((dataset["label"], torch.zeros(local_data.shape[0]) + i / args.continuation_size))
        dataset["embedding"] = torch.cat((dataset["embedding"], local_embedding))
        dataset["prediction"] = torch.cat((dataset["prediction"], local_memorized))
        dataset["entropy"] = torch.cat((dataset["entropy"], local_entropy))
    print("data load finished")
    dataset = Dataset.from_dict(dataset)
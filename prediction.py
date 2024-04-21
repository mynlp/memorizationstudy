import numpy as np
import torch
import random
from datasets import Dataset
from transformers import GPTNeoXForCausalLM
from tqdm import tqdm
from models import *
from torch.utils.data import DataLoader
import argparse
import os

def format_example(example):
    tokens, labels, embeddings = example['token'], example['label'], example["embedding"]
    return {'input_ids': tokens, 'labels': labels, 'embedding': embeddings}

def evaluate(predictor, dataloader, counter=0):
    predictor.eval()  # Set the model to evaluation mode
    total_loss = 0
    data_size = 0
    with torch.no_grad():  # Do not calculate gradient since we are only evaluating
        for data in dataloader:
            data_size += len(data["labels"])
            embedding = torch.stack([torch.stack(x, dim=1) for x in data["embedding"]], dim=1)
            scores_mean, standard = infer(predictor, embedding)
            loss = loss_fn(scores_mean.squeeze(), data["labels"].float().to(device))
            in_range = ((data["labels"].float() > (scores_mean - standard)) &
                        (data["labels"].float() < (scores_mean + standard))).float()
            counter += torch.sum(in_range).item()
            total_loss += loss.item()
    return total_loss / len(dataloader), counter/data_size

def infer(predictor, embeddings, repeats=10):
    predictor.eval()  # Set the model to evaluation mode
    scores_list = []
    with torch.no_grad():  # Do not calculate gradient since we are only inferring
        for _ in range(repeats):
            scores = predictor.infer(embeddings.float().cuda())
            scores_list.append(scores.squeeze())
    scores_arr = torch.tensor(scores_list)
    return scores_arr.mean(), scores_arr.var()

args = argparse.ArgumentParser()
args.add_argument("--model_size", type=str, default="70m")
args.add_argument("--context", type=int, default=32)
args.add_argument("--continuation", type=int, default=16)
args.add_argument("--checkpoint", type=int, default=143000)
args.add_argument("--seed", type=int, default=42)
args = args.parse_args()
random.seed(args.seed)
model_size = "70m"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = f"EleutherAI/pythia-{args.model_size}-deduped-v0"
context = 32
from datasets import DatasetDict
dataset = {"token": [], "label": [], "embedding": []}
for i in range(args.continuation):
    local_data = torch.load(f"cross_remembered/context_tokens_{args.continuation}_{i}_{model_size}.pt")
    local_embedding = torch.load(f"cross_remembered/embeddings_{args.continuation}_{i}_{model_size}.pt")
    dataset["token"].append(local_data)
    dataset["label"].append(torch.zeros(local_data.shape[0])+ i/args.continuation)
    dataset["embedding"].append(local_embedding)
model = GPTNeoXForCausalLM.from_pretrained(
        model_name,
        use_cache=False,
        revision=f'step143000',
    ).eval()
# model = model.to_bettertransformer()
# model = model.to(device)
model.generation_config.pad_token_id = model.generation_config.eos_token_id
model.generation_config.output_hidden_states = True
model.generation_config.output_attentions = True
model.generation_config.output_scores = True
model.generation_config.return_dict_in_generate = True

dataset["token"] = torch.cat(dataset["token"])
dataset["label"] = torch.cat(dataset["label"])
dataset["embedding"] = torch.cat(dataset["embedding"])
dataset = Dataset.from_dict(dataset)
splited_dataset = dataset.train_test_split(test_size=0.2)
embedding_size = model.config.hidden_size
hidden_size = 64  # You can choose the hidden size according to your needs
predictor = Predictor(embedding_size, hidden_size).to(device)

# Define a loss function and an optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(predictor.parameters())
train_dataset = splited_dataset['train']
if f"{args.model_size}.arrow" not in os.listdir("train_cache"):
    train_dataset = train_dataset.map(format_example, batched=True, num_proc=8, cache_file_name=f"train_cache/{args.model_size}.arrow")
else:
    train_dataset = Dataset.load_from_disk(f"train_cache/{args.model_size}.arrow")
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32)

# Prepare test dataloader
test_dataset = splited_dataset['test']
if f"{args.model_size}.arrow" not in os.listdir("test_cache"):
    test_dataset = test_dataset.map(format_example, batched=True, num_proc=8, cache_file_name=f"test_cache/{args.model_size}.arrow")
else:
    test_dataset = Dataset.load_from_disk(f"test_cache/{args.model_size}.arrow")
test_dataloader = DataLoader(test_dataset, batch_size=32)

# Training loop
for i, data in tqdm(enumerate(train_dataloader)):
    embedding = torch.stack([torch.stack(x, dim=1) for x in data["embedding"]], dim=1)
    scores = predictor(embedding.float().cuda())
        # Compute the loss
    loss = loss_fn(scores.squeeze(), data["labels"].float().to(device))
    # Backprop and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
validation_loss, accuracy = evaluate(predictor, test_dataloader, )
print(f'Validation Loss: {validation_loss:.4f}')
print(f'Accuracy: {accuracy:.4f}')

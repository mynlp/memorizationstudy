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

def evaluate(predictor, dataloader):
    predictor.eval()  # Set the model to evaluation mode
    total_loss = 0
    data_size = 0
    counter = 0
    with torch.no_grad():  # Do not calculate gradient since we are only evaluating
        for data in dataloader:
            embedding = torch.stack([torch.stack(x, dim=1) for x in data["embedding"]], dim=1)
            entropy = torch.stack([x for x in data["entropy"]], dim=1)
            prediction = torch.stack([x for x in data["prediction"]], dim=1)
            classes = infer(predictor, embedding, entropy)
            classification_loss = classification_loss_fn(classes.squeeze().view(-1, 2),
                                                         prediction.type(torch.int64).view(-1).to(device))
            data_size = prediction.shape[0]*prediction.shape[1]
            pdb.set_trace()
            classificaiton_results = classes.squeeze().view(-1, 2).argmax(dim=2) == prediction.type(torch.int64).view(-1).to(device)
            classificaiton_results = classificaiton_results.float().sum()
            loss = classification_loss
            total_loss += loss.item()
            counter += classificaiton_results
    return total_loss / len(dataloader), counter/data_size
    #         scores_mean, standard = infer(predictor, embedding)
    #         loss = loss_fn(scores_mean.squeeze(), data["labels"].float().to(device))
    #         in_range = ((data["labels"].float().cuda() > (scores_mean - 3*standard)) &(data["labels"].float().cuda() < (scores_mean + 3*standard))).float()
    #         counter += torch.sum(in_range).item()
    #         total_loss += loss.item()
    # return total_loss / len(dataloader), counter/data_size

def infer(predictor, embeddings, entropy, repeats=50):
    predictor.eval()  # Set the model to evaluation mode
    scores_list = []
    with torch.no_grad():  # Do not calculate gradient since we are only inferring
        #for _ in range(repeats):
        classes = predictor(embeddings.float().cuda(), entropy.float().cuda())
    return classes
        #scores, classes = predictor.infer(embeddings.float().cuda())
        #scores_list.append(scores.squeeze())
    #scores_arr = torch.stack(scores_list, dim=1)
    #return scores_arr.mean(dim=1), scores_arr.var(dim=1)

args = argparse.ArgumentParser()
args.add_argument("--model_size", type=str, default="70m")
args.add_argument("--context", type=int, default=128)
args.add_argument("--continuation", type=int, default=16)
args.add_argument("--checkpoint", type=int, default=143000)
args.add_argument("--seed", type=int, default=42)
args.add_argument("--epoch", type=int, default=20)
args.add_argument("--embedding_size", type=int, default=512)
args.add_argument("--hidden_size", type=int, default=512)
args.add_argument("--load_cache", type=bool, default=True)
args.add_argument("--model_type", type=str, default="lstm")
args.add_argument("--batch_size", type=int, default=128)
args = args.parse_args()
random.seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
model_name = f"EleutherAI/pythia-{args.model_size}-deduped-v0"
if num_gpus > 1:
    print("Number of available GPUs: ", num_gpus)
else:
    print("Number of available GPU: ", num_gpus)
if args.load_cache == False:
    dataset = {"token": [], "label": [], "embedding": [], "prediction":[], "entropy":[]}
    for i in range(args.continuation):
        local_data = torch.load(f"cross_remembered/context_tokens_{args.continuation}_{i}_{args.model_size}.pt", map_location=device)
        local_embedding = torch.load(f"cross_remembered/embeddings_{args.continuation}_{i}_{args.model_size}.pt", map_location=device)
        local_entropy = torch.load(f"cross_remembered/entropy_{args.continuation}_{i}_{args.model_size}.pt", map_location=device)
        local_memorized = torch.load(f"cross_remembered/memorized_idx_{args.continuation}_{i}_{args.model_size}.pt", map_location=device)
        dataset["token"].append(local_data)
        dataset["label"].append(torch.zeros(local_data.shape[0])+ i/args.continuation)
        dataset["embedding"].append(local_embedding)
        dataset["prediction"].append(local_memorized)
        dataset["entropy"].append(local_entropy)
    dataset["token"] = torch.cat(dataset["token"])
    dataset["label"] = torch.cat(dataset["label"])
    dataset["embedding"] = torch.cat(dataset["embedding"])
    dataset["prediction"] = torch.cat(dataset["prediction"])
    dataset["entropy"] = torch.cat(dataset["entropy"])
    dataset = Dataset.from_dict(dataset)
    splited_dataset = dataset.train_test_split(test_size=0.2)
    train_dataset = splited_dataset['train']
    test_dataset = splited_dataset['test']
    train_dataset = train_dataset.map(format_example, batched=True,  cache_file_name=f"train_cache/{args.model_size}.arrow")
    test_dataset = test_dataset.map(format_example, batched=True, cache_file_name=f"test_cache/{args.model_size}.arrow")
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
else:
    train_dataset = Dataset.from_file(f"train_cache/{args.model_size}.arrow")
    test_dataset = Dataset.from_file(f"test_cache/{args.model_size}.arrow")
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

if args.model_type == "lstm":
    predictor = LSTMPredictor(args.embedding_size, args.hidden_size)
elif args.model_type == "transformer":
    predictor = TransformerPredictor(args.embedding_size, args.hidden_size)
if num_gpus > 1:
    predictor = nn.DataParallel(predictor)
predictor.to(device)# Define a loss function and an optimizer
#loss_fn = nn.MSELoss()
classification_loss_fn = nn.NLLLoss()
optimizer = torch.optim.Adam(predictor.parameters(), lr=0.001)

train_loss = []
best_accuracy = 0
best_model_state = None
accumulated_loss = 0
# Training loop
for _ in range(args.epoch):
    predictor.train()
    #evaluate(predictor, test_dataloader)
    for i, data in tqdm(enumerate(train_dataloader)):
        predictor.zero_grad()
        embedding = torch.stack([torch.stack(x, dim=1) for x in data["embedding"]], dim=1)
        entropy = torch.stack([x for x in data["entropy"]], dim=1)
        prediction = torch.stack([x for x in data["prediction"]], dim=1)
        classes = predictor(embedding.float().cuda(), entropy.float().cuda())
            # Compute the loss
        loss = classification_loss_fn(classes.squeeze().view(-1, 2), prediction.type(torch.int64).view(-1).to(device))
        # Backprop and optimize
        loss.backward()
        accumulated_loss += loss.item()
        train_loss.append(loss.item())
        optimizer.step()
        if i % 100 == 0:
            print(f'Loss: {accumulated_loss/100:.4f}')
            accumulated_loss = 0
    predictor.eval()
    validation_loss, accuracy = evaluate(predictor, test_dataloader)
    print(f'Validation Loss: {validation_loss:.4f}')
    print(f'Accuracy: {accuracy:.4f}')
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_state = predictor.state_dict()
torch.save(best_model_state, f"saved_models/predictor_{args.model_size}.pt")
plt.plot(train_loss)
plt.savefig(f"prediction_train_loss_{args.model_size}.png")
plt.show()


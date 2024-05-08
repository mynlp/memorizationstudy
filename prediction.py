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
    token_data_size = 0
    row_data_size = 0
    counter = 0
    full＿acc_counter = 0
    with torch.no_grad():  # Do not calculate gradient since we are only evaluating
        for data in dataloader:
            embedding = torch.stack([torch.stack(x, dim=1) for x in data["embedding"]], dim=1)
            entropy = torch.stack([x for x in data["entropy"]], dim=1)
            prediction = torch.stack([x for x in data["prediction"]], dim=1)
            classes = infer(predictor, embedding, entropy)
            classification_loss = classification_loss_fn(classes.squeeze().view(-1, 2),
                                                         prediction.type(torch.int64).view(-1).to(device))
            row_data_size += prediction.shape[0]
            token_data_size += prediction.shape[0]*prediction.shape[1]
            #pdb.set_trace()
            classificaiton_results = classes.squeeze().argmax(dim=2) == prediction.type(torch.int64).to(device)
            row_eq_res = torch.all(classificaiton_results, dim=1)
            classificaiton_results = classificaiton_results.float().sum()
            loss = classification_loss
            total_loss += loss.item()
            counter += classificaiton_results
            full＿acc_counter += row_eq_res.float().sum()
    return total_loss / len(dataloader), counter/token_data_size, full＿acc_counter/row_data_size


def infer(predictor, embeddings, entropy, repeats=50):
    predictor.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Do not calculate gradient since we are only inferring
        classes = predictor(embeddings.float().cuda(), entropy.float().cuda())
    return classes


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
    dataset = Dataset.from_dict(dataset)
    splited_dataset = dataset.train_test_split(test_size=0.2)
    train_dataset = splited_dataset['train']
    test_dataset = splited_dataset['test']
    train_dataset = train_dataset.map(format_example, batched=True,  cache_file_name=f"train_cache/{args.model_size}_{args.context_size}_{args.continuation_size}.arrow")
    test_dataset = test_dataset.map(format_example, batched=True, cache_file_name=f"test_cache/{args.model_size}_{args.context_size}_{args.continuation_size}.arrow")
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
else:
    train_dataset = Dataset.from_file(f"train_cache/{args.model_size}_{args.context_size}_{args.continuation_size}.arrow")
    test_dataset = Dataset.from_file(f"test_cache/{args.model_size}_{args.context_size}_{args.continuation_size}.arrow")
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

if args.model_type == "lstm":
    predictor = LSTMPredictor(embedding_size, args.hidden_size)
elif args.model_type == "transformer":
    predictor = TransformerPredictor(embedding_size, args.hidden_size)
if num_gpus > 1:
    predictor = nn.DataParallel(predictor)
predictor.to(device)# Define a loss function and an optimizer
#loss_fn = nn.MSELoss()
classification_loss_fn = nn.NLLLoss()
optimizer = torch.optim.Adam(predictor.parameters(), lr=args.lr)

train_loss = []
best_accuracy = 0
best_full_accuracy = 0
best_model_state = None
accumulated_loss = 0
f = open(f"prediction_result/{args.model_size}_{args.context_size}_{args.continuation_size}.txt", "w")
early_stop_counter = 0
prev_accuracy = 0
# Training loop
for _ in range(args.epoch):
    if early_stop_counter >= 5:  # Stop training if accuracy has decreased 5 times in a row
        print("Early stopping")
        break
    evaluate(predictor, test_dataloader)
    predictor.train()
    for i, data in tqdm(enumerate(train_dataloader)):
        predictor.zero_grad()
        embedding = torch.stack([torch.stack(x, dim=1) for x in data["embedding"]], dim=1).cuda()
        entropy = torch.stack([x for x in data["entropy"]], dim=1).cuda()
        prediction = torch.stack([x for x in data["prediction"]], dim=1).cuda()
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
    validation_loss, accuracy, full_acc = evaluate(predictor, test_dataloader)
    print(f'Validation Loss: {validation_loss:.4f}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Full Accuracy: {full_acc:.4f}')
    f.write(f'Validation Loss: {validation_loss:.4f}\n')
    f.write(f'Accuracy: {accuracy:.4f}\n')
    f.write(f'Full Accuracy: {full_acc:.4f}\n')
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_full_accuracy = full_acc
        best_model_state = predictor.state_dict()
        early_stop_counter = 0
    elif accuracy < prev_accuracy:
        early_stop_counter += 1  # Increase counter if accuracy decreased
    prev_accuracy = accuracy
print(f'Best Accuracy: {best_accuracy:.4f}')
print(f'Best Full Accuracy: {best_full_accuracy:.4f}')
f.write(f'Best Accuracy: {best_accuracy:.4f}\n')
f.write(f'Best Full Accuracy: {best_full_accuracy:.4f}\n')
f.close()
torch.save(best_model_state, f"saved_models/predictor_{args.model_size}_{args.model_type}_{args.context_size}_{args.continuation_size}.pt")
plt.plot(train_loss)
plt.savefig(f"prediction_train_loss_{args.model_size}_{args.model_type}_{args.context_size}_{args.continuation_size}.png")
plt.show()



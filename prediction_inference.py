import pdb
from matplotlib import pyplot as plt
import numpy as np
import torch
import random
from datasets import Dataset
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from tqdm import tqdm
from models import *
from torch.utils.data import DataLoader
import argparse
import pdb
import os
from datasets import DatasetDict
import pdb


def format_example(example):
    tokens, labels, embeddings, prediction, entropy = example['token'], example['label'], example["embedding"], example[
        "prediction"], example["entropy"]
    return {'input_ids': tokens, 'labels': labels, 'embedding': embeddings, 'prediction': prediction,
            'entropy': entropy}


def output_probability(idx, tokens, probs, prediction, tokenizer, classificaiton_results, predcited, f):
    print("\n###Begining of Sentence###\n")
    f.write("\n###Begining of Sentence###\n")
    sent_token = [tokenizer.decode(token_id) for token_id in tokens[idx]]
    succeded["tokens"].append(sent_token)
    succeded["probability"].append(probs)
    for sent_idx, token in enumerate(sent_token[args.context_size:]):
        if predcited[idx][sent_idx] == 1:
            print(f"Memorized Probability of token {token} at {idx}:{probs[idx][sent_idx][1]}")
            f.write(f"Memorized Probability of token {token} at {idx}:{probs[idx][sent_idx][1]}\n")
            if classificaiton_results[idx][sent_idx] == 0:
                print("The actual label should be unmemorized")
                f.write("The actual label should be unmemorized\n")
        else:
            print(f"Unmemorized Probability of token {token} at {idx}:{probs[idx][sent_idx][0]}")
            f.write(f"Unmemorized Probability of token {token} at {idx}:{probs[idx][sent_idx][0]}\n")
            if classificaiton_results[idx][sent_idx] == 0:
                print("The actual label should be memorized")
                f.write("The actual label should be memorized\n")
    print(tokenizer.decode(tokens[idx]))
    print("\n###End of Sentence###\n")
    f.write(tokenizer.decode(tokens[idx]) + "\n")


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
            token_data_size += prediction.shape[0] * prediction.shape[1]
            #pdb.set_trace()
            classificaiton_results = classes.squeeze().argmax(dim=2) == prediction.type(torch.int64).to(device)
            row_eq_res = torch.all(classificaiton_results, dim=1)
            classificaiton_results = classificaiton_results.float().sum()
            loss = classification_loss
            total_loss += loss.item()
            counter += classificaiton_results
            full＿acc_counter += row_eq_res.float().sum()
    return total_loss / len(dataloader), counter / token_data_size, full＿acc_counter / row_data_size


def infer(predictor, embeddings, entropy, repeats=50):
    predictor.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Do not calculate gradient since we are only inferring
        classes = predictor(embeddings.float().cuda(), entropy.float().cuda())
    return classes


args = argparse.ArgumentParser()
args.add_argument("--model_size", type=str, default="410m")
args.add_argument("--context_size", type=int, default=32)
args.add_argument("--continuation_size", type=int, default=32)
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

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    revision=f"step{143000}",
    cache_dir=f"./pythia-{args.model_size}-deduped/step{143000}",
)
if num_gpus > 1:
    print("Number of available GPUs: ", num_gpus)
else:
    print("Number of available GPU: ", num_gpus)

#train_dataset = Dataset.from_file(f"train_cache/{args.model_size}_{args.context_size}_{args.continuation_size}.arrow")
test_dataset = Dataset.from_file(f"test_cache/{args.model_size}_{args.context_size}_{args.continuation_size}.arrow")
#train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

if args.model_type == "lstm":
    predictor = LSTMPredictor(embedding_size, args.hidden_size)
elif args.model_type == "transformer":
    predictor = TransformerPredictor(embedding_size, args.hidden_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() >= 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    predictor = nn.DataParallel(predictor)
predictor.load_state_dict(torch.load(
    f"saved_models/predictor_{args.model_size}_{args.model_type}_{args.context_size}_{args.continuation_size}.pt",
    map_location=device))
predictor = predictor.to(device)

predictor.eval()  # Set the model to evaluation mode
total_loss = 0
token_data_size = 0
row_data_size = 0
counter = 0
full＿acc_counter = 0
succeded = {"tokens": [], "probability": []}
memorized_dict = {}
f = open(f"visulization_{args.model_size}.txt", "w")
with torch.no_grad():  # Do not calculate gradient since we are only evaluating
    for data in test_dataloader:
        embedding = torch.stack([torch.stack(x, dim=1) for x in data["embedding"]], dim=1)
        tokens = torch.stack([x for x in data["token"]], dim=1)
        entropy = torch.stack([x for x in data["entropy"]], dim=1)
        prediction = torch.stack([x for x in data["prediction"]], dim=1)
        classes = predictor(embedding.float().cuda(), entropy.float().cuda())
        probs = torch.exp(classes)
        classificaiton_results = classes.squeeze().argmax(dim=2) == prediction.type(torch.int64).to(device)
        row_eq_res = torch.all(classificaiton_results, dim=1)
        mem_score = prediction.sum(dim=1) / 16
        for idx, score in enumerate(classificaiton_results.sum(dim=1) / 16):
            if score == 1:
                print("Prediction Score: 1")
                print(f"Memorization Score:{mem_score[idx]}")
                f.write(f"Prediction Score: 1\n")
                f.write(f"Memorization Score:{mem_score[idx]}\n")
                if mem_score[idx].item() not in memorized_dict:
                    memorized_dict[mem_score[idx].item()] = 1
                else:
                    memorized_dict[mem_score[idx].item()] += 1
                output_probability(idx, tokens, probs, prediction, tokenizer, classificaiton_results,
                                   classes.squeeze().argmax(dim=2), f)
            elif score == 0.5:
                print("Prediction Score: 0.5")
                print(f"Memorization Score:{mem_score[idx]}")
                f.write(f"Prediction Score: 0.5\n")
                f.write(f"Memorization Score:{mem_score[idx]}\n")
                output_probability(idx, tokens, probs, prediction, tokenizer, classificaiton_results,
                                   classes.squeeze().argmax(dim=2), f)
        #classificaiton_results = classificaiton_results.float().sum()
f.close()
print(memorized_dict)
prediction_results={0.0625: 54, 0.25: 13, 0.375: 16, 0.9375: 27, 0.6875: 22, 0.0: 105, 0.8125: 19, 0.4375: 12, 0.625: 20, 0.75: 18, 0.1875: 25, 0.875: 25, 0.5625: 9, 0.125: 30, 0.5: 20, 0.3125: 15, 1.0: 1}
sorted_keys = sorted(prediction_results)
sorted_values = [prediction_results[key] for key in sorted_keys]

plt.figure(figsize=[10, 8])
plt.bar(range(len(prediction_results)), sorted_values, tick_label=sorted_keys, color='skyblue', alpha=1)
plt.xlabel('Score', fontsize=16)
plt.xticks(rotation=90)
plt.ylabel('Count', fontsize=16)
plt.title('Full Accuracy Distribution over Memorization Score on 2.8b Model', fontsize=18)
plt.grid(True)
plt.tight_layout()
plt.show()
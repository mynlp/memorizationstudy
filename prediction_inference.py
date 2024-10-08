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
args.add_argument("--model_size", type=str, default="6.9b")
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

prediction_results_2_8b = {0.0625: 54, 0.25: 13, 0.375: 16, 0.9375: 27, 0.6875: 22, 0.0: 105, 0.8125: 19, 0.4375: 12,
                      0.625: 20, 0.75: 18, 0.1875: 25, 0.875: 25, 0.5625: 9, 0.125: 30, 0.5: 20, 0.3125: 15, 1.0: 1}
prediction_results_1b = {0.75: 19, 0.1875: 17, 0.5625: 11, 0.0625: 60, 0.0: 93, 0.3125: 21, 0.9375: 22, 0.125: 28,
                         0.4375: 17, 0.875: 29, 0.25: 19, 0.8125: 24, 0.5: 21, 0.625: 19, 0.375: 25, 0.6875: 18, 1.0: 2}
prediction_results_70m = {0.5625: 20, 0.8125: 36, 0.0: 131, 0.875: 45, 0.3125: 16, 0.25: 17, 0.75: 35, 0.625: 38,
                          0.0625: 76, 0.4375: 20, 0.375: 21, 0.125: 45, 0.6875: 27, 0.5: 24, 0.9375: 59, 1.0: 9,
                          0.1875: 18}
prediction_results_6_9b={0.0: 219, 0.0625: 41, 0.4375: 18, 0.875: 14, 0.5: 17, 0.9375: 24, 0.375: 14, 0.625: 18, 0.5625: 23, 0.75: 12, 0.25: 9, 0.1875: 16, 0.125: 18, 0.6875: 18, 0.3125: 10, 0.8125: 16, 0.875:0, 0.9375:0, 1.0:0}
sorted_keys = sorted(prediction_results_2_8b)

# Normalize the results
total_1 = sum(prediction_results_2_8b.values())
total_2 = sum(prediction_results_1b.values())
total_3 = sum(prediction_results_70m.values())
total_4 = sum(prediction_results_6_9b.values())

sorted_values_1 = [prediction_results_2_8b[key] / total_1 for key in sorted_keys]
sorted_values_2 = [prediction_results_1b[key] / total_2 for key in sorted_keys]
sorted_values_3 = [prediction_results_70m[key] / total_3 for key in sorted_keys]
sorted_values_4 = [prediction_results_6_9b[key] / total_4 for key in sorted_keys]

# Create bar width
bar_width = 0.2
index = np.arange(len(prediction_results_2_8b))

plt.figure(figsize=[12, 8])

plt.bar(index, sorted_values_3, bar_width, color='skyblue', alpha=1, label='70m')
plt.bar(index + bar_width, sorted_values_2, bar_width, color='orange', alpha=1, label='1b')
plt.bar(index + 2 * bar_width, sorted_values_1, bar_width, color='green', alpha=1, label='2.8b')
plt.bar(index + 3 * bar_width, sorted_values_4, bar_width, color='red',alpha=1, label='6.9b')

plt.xlabel('Memorization Score', fontsize=16)
plt.xticks(index + bar_width * 3, sorted_keys, rotation=90, fontsize=14)  # reposition x-ticks location
plt.ylabel('Proportion', fontsize=16)  # Change y label to "Proportion"
plt.title('Normalized Full Accuracy Count Distribution over Memorization Score',
          fontsize=18)  # Change title to indicate it's normalized
plt.legend(fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.savefig('normalized_full_accuracy_count_distribution.png', dpi=600)
plt.show()

import numpy as np
import torch
import random
from datasets import Dataset
from transformers import GPTNeoXForCausalLM
from tqdm import tqdm
from models import *
from torch.utils.data import DataLoader

def format_example(example):
    tokens, labels = example['token'], example['label']
    return {'input_ids': tokens, 'labels': labels}

random.seed(42)
model_size = "70m"
model_name = f"EleutherAI/pythia-{model_size}-deduped-v0"
CHECKPOINT = 143000
context = 32
continuation = 16
from datasets import DatasetDict
dataset = {"token": [], "label": []}
for i in range(continuation):
    local_data = torch.load(f"cross_remembered/context_tokens_{continuation}_{i}_{model_size}.pt")
    dataset["token"].append(local_data)
    dataset["label"].append(torch.zeros(local_data.shape[0])+ i/continuation)
model = GPTNeoXForCausalLM.from_pretrained(
        model_name,
        use_cache=False,
        revision=f'step143000',
    ).half().eval()
model = model.to_bettertransformer()
model = model.cuda()
model.generation_config.pad_token_id = model.generation_config.eos_token_id
model.generation_config.output_hidden_states = True
model.generation_config.output_attentions = True
model.generation_config.output_scores = True
model.generation_config.return_dict_in_generate = True

dataset["token"] = torch.cat(dataset["token"])
dataset["label"] = torch.cat(dataset["label"])
dataset = Dataset.from_dict(dataset)
splited_dataset = dataset.train_test_split(test_size=0.2)
embedding_size = model.config.n_embd
hidden_size = 64  # You can choose the hidden size according to your needs
predictor = Predictor(embedding_size, hidden_size)

# Define a loss function and an optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(predictor.parameters())
train_dataset = splited_dataset['train']
train_dataloader = DataLoader(train_dataset.map(format_example), shuffle=True, batch_size=32)

# Prepare test dataloader
test_dataset = splited_dataset['test']
test_dataloader = DataLoader(test_dataset.map(format_example), batch_size=32)

# Training loop
for i, (tokens, labels) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
    with torch.no_grad():
        model_outputs = model.generate(tokens[:, :context], temperature=0.0, top_k=0, top_p=0,
                                       max_length=context + continuation, min_length=context + continuation)
        context_embedding = model_outputs.hidden_states[0][-1]
        embedding = torch.stack([x[-1] for x in model_outputs.hidden_states[1:]]).mean(0).squeeze()
        embeddings = torch.concat([context_embedding, embedding], dim=1)
    # Forward pass through the predictor
    scores = predictor(embeddings)

    # Compute the loss
    loss = loss_fn(scores.squeeze(), labels.float())

    # Backprop and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
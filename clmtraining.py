import datasets
from datasets import load_dataset, DatasetDict,Dataset
from transformers import AutoTokenizer, AutoConfig
from transformers import DataCollatorForLanguageModeling, GPT2LMHeadModel
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
import math
import torch
import pdb
from tqdm import tqdm
from collections import defaultdict



def tokenize(element):
    """
    Tokenizes the given element.

    :param element: The element to be tokenized.
    :return: A dictionary containing the batch of input_ids.
    """
    cache = []
    for sample in element["text"]:
        if sample is not None:
            cache.append(sample)
    print(len(cache))
    outputs = tokenizer(
        cache,
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}

def filter_dataset(dataset):
    filtered_dict = defaultdict(list)
    for sample in tqdm(dataset["train"]["text"]):
        if sample["text"] is not None:
            for k, v in sample.items():
                filtered_dict[k].append(v)
    return Dataset.from_dict(filtered_dict)

#raw_dataset = Dataset.from_dict({"input_ids": torch.load("cross_remembered/context_tokens.pt").view(-1,2049)})
raw_dataset = datasets.load_dataset("json", data_files="cross_remembered/memorized_text.json")
#processed_raw_dataset = filter_dataset(raw_dataset)
context_length = 512
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
tokenized_datasets = raw_dataset.map(
    tokenize, batched=True, remove_columns=raw_dataset["train"].column_names
)
tokenized_datasets = tokenized_datasets["train"].train_test_split(test_size=0.2)

CHECKPOINT = 143000
config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),
    n_ctx=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
model = GPT2LMHeadModel(config)
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
args = TrainingArguments(
    output_dir="clm_output",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    evaluation_strategy="steps",
    eval_steps=5_000,
    logging_steps=5_000,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=5_000,
    fp16=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

trainer.train()
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")


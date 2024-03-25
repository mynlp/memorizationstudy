from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
import math
import torch

def preprocess_function(examples):
    return tokenizer([" ".join(x) for x in examples["answers.text"]])

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

data = torch.load("cross_remembered/context_tokens.pt")
eli5 = load_dataset("eli5", split="train_asks[:5000]")
eli5 = eli5.train_test_split(test_size=0.2)
model_name = "EleutherAI/pythia-160m-deduped-v0"
CHECKPOINT= 143000
tokenizer = AutoTokenizer.from_pretrained(
  model_name,
  revision=f"step{CHECKPOINT}",
  cache_dir=f"./pythia-160m-deduped/step{CHECKPOINT}",
)
eli5 = eli5.flatten()
tokenized_eli5 = eli5.map(
    preprocess_function,
    batched=True,
    remove_columns=eli5["train"].column_names,
)
block_size = 128
lm_dataset = tokenized_eli5.map(group_texts, batched=True)
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
training_args = TrainingArguments(
    output_dir="small_demo",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["test"],
    data_collator=data_collator,
)

trainer.train()
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

from datasets import load_dataset, DatasetDict,Dataset
from transformers import AutoTokenizer, AutoConfig
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
import math
import torch


def batchfy(data):
  input_batch = []
  for input_ids in (data):
      input_batch.append(input_ids)
  return {"input_ids": input_batch}

ds = Dataset.from_dict({"input_ids": torch.load("cross_remembered/context_tokens.pt").view(-1,2049)})
model_name = "EleutherAI/pythia-160m-deduped-v0"
CHECKPOINT = 143000
context_length = 128
tokenizer = AutoTokenizer.from_pretrained(
  model_name,
  revision=f"step{CHECKPOINT}",
  cache_dir=f"./pythia-160m-deduped/step{CHECKPOINT}",
)
config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),
    n_ctx=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
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

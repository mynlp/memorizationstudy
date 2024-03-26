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

raw_dataset = Dataset.from_dict({"input_ids": torch.load("cross_remembered/context_tokens.pt").view(-1,2049)})
model_name = "EleutherAI/pythia-160m-deduped-v0"
CHECKPOINT = 143000
context_length = 128
tokenizer = AutoTokenizer.from_pretrained(
  model_name,
  revision=f"step{CHECKPOINT}",
  cache_dir=f"./pythia-160m-deduped/step{CHECKPOINT}",
)
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),
    n_ctx=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)


raw_dataset = raw_dataset.flatten()
lm_dataset = raw_dataset.map(
    batchfy,
    batched=True,
)
block_size = 128
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
args = TrainingArguments(
    output_dir="codeparrot-ds",
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
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["valid"],
)

trainer.train()
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
from utils import InstructDataset,InstructCollator
from torch.utils.data import DataLoader
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
from transformers import Trainer, TrainingArguments
import argparse

# model = AutoModelForCausalLM.from_pretrained(f"EleutherAI/pythia-160m-deduped-v0",
#                                              revision="step143000", cache_dir=f"./pythia-160m-deduped/step143000")
# tokenizer = AutoTokenizer.from_pretrained(f"EleutherAI/pythia-160m-deduped-v0")
paser = argparse.ArgumentParser()
paser.add_argument("--batch_size", type=int, default=256)
paser.add_argument("--context_size", type=int, default=32)
paser.add_argument("--continuation_size", type=int, default=16)
paser.add_argument("--model", type=str, default="70m-deduped-v0")
paser.add_argument("--checkpoint", type=int, default=143000)
paser.add_argument("--rank", type=int, default=0)
args = paser.parse_args()
model = AutoModelForCausalLM.from_pretrained(f"EleutherAI/pythia-{args.model}", revision=f"step{args.checkpoint}",
                                             device_map="auto", torch_dtype=torch.float16)
#tokenizer = AutoTokenizer.from_pretrained("cyberagent/open-calm-7b")
tokenizer = AutoTokenizer.from_pretrained(
  f"EleutherAI/pythia-{args.model}",
  revision=f"step{args.checkpoint}",
  cache_dir=f"./pythia-{args.model}/step{args.checkpoint}",
)
dolly_ja = datasets.load_dataset("databricks/databricks-dolly-15k")
PROMPT_DICT = {
    "prompt_input": (
        "The following is the combination of instruction that explains the task and the context of the input."
        "Please response with answer that meets the requirements。\n\n"
        "### Instruction:\n{instruction}\n\n### Context:{context}\n\n### Answer:"
    ),
    "prompt_no_input": (
        "The following is the instruction that explains the task."
        "Please response with answer that meets the requirements.\n\n"
        "### Instruction:\n{instruction}\n\n### Answer:"
    )
}
train_dataset = InstructDataset(dolly_ja, tokenizer, PROMPT_DICT)
collator = InstructCollator(tokenizer)
loader = DataLoader(train_dataset, collate_fn=collator, batch_size=8, shuffle=True)

for param in model.parameters():
    param.requires_grad = False # モデルをフリーズ
    if param.ndim == 1:
        # 安定のためにレイヤーノルムをfp32にキャスト
        param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()
model.enable_input_require_grads()

class CastOutputToFloat(nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)
model.embed_out = CastOutputToFloat(model.embed_out)
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    fan_in_fan_out=False,
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
training_args = TrainingArguments(
        output_dir='./output',
        save_total_limit=1,
        per_device_train_batch_size=8,
        num_train_epochs=1,
        remove_unused_columns=False,
        logging_steps=20,
        fp16=True,
        dataloader_num_workers=16,
        report_to="none",
)

trainer = Trainer(
        model=model,
        data_collator=collator,
        args=training_args,
        train_dataset=train_dataset,
    )

model.config.use_cache = False
trainer.train()
model.save_pretrained("./instruction_tuned_pythia")

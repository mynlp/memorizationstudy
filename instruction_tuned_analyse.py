import os
import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from peft import PeftModel, PeftConfig

model_path = "cyberagent/open-calm-7b"
base_llm1 = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model1 = PeftModel.from_pretrained(base_llm1, './output1', torch_dtype=torch.float16)
model1.half()

PROMPT_DICT = {
    "prompt_input": (
        "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。"
        "要求を適切に満たす応答を書きなさい。\n\n"
        "### 指示:\n{instruction}\n\n### 入力:{input}\n\n### 応答:"
    ),
    "prompt_no_input": (
        "以下は、タスクを説明する指示です。"
        "要求を適切に満たす応答を書きなさい。\n\n"
        "### 指示:\n{instruction}\n\n### 応答:"
    )
}

if temp < 0.1:
    temp = 0.1
elif temp > 1:
    temp = 1

input_prompt = {
    'instruction': instruction,
    'input': conversation
}

prompt = ''
if input_prompt['input'] == '':
    prompt = PROMPT_DICT['prompt_no_input'].format_map(input_prompt)
else:
    prompt = PROMPT_DICT['prompt_input'].format_map(input_prompt)

inputs = tokenizer(
    prompt,
    return_tensors='pt'
).to(model.device)

with torch.no_grad():
    tokens = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=temp,
        top_p=0.9,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.pad_token_id,
    )

output = tokenizer.decode(tokens[0], skip_special_tokens=True)
output = output.split('\n\n')[-1].split(':')[-1]
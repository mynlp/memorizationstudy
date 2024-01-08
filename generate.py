from transformers import GPTNeoXForCausalLM, AutoTokenizer,  AutoModelForCausalLM,  LogitsProcessorList, MinLengthLogitsProcessor, StoppingCriteriaList,  MaxLengthCriteria
model = GPTNeoXForCausalLM.from_pretrained(
  "EleutherAI/pythia-70m-deduped",
  revision="step143000",
  cache_dir="./pythia-70m-deduped/step143000",
)

tokenizer = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-70m-deduped",
  revision="step143000",
  cache_dir="./pythia-70m-deduped/step143000",
)

#tokenizer = AutoTokenizer.from_pretrained("gpt2")
#model = AutoModelForCausalLM.from_pretrained("gpt2")

# set pad_token_id to eos_token_id because GPT2 does not have a PAD token
model.generation_config.pad_token_id = model.generation_config.eos_token_id
model.generation_config.output_hidden_states = True
model.generation_config.output_attentions = True
model.generation_config.output_scores = True
model.generation_config.return_dict_in_generate = True
input_prompt = "It might be possible to"
input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

# instantiate logits processors
logits_processor = LogitsProcessorList(
[
MinLengthLogitsProcessor(10, eos_token_id=model.generation_config.eos_token_id),
]
)
stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

outputs = model.greedy_search(
input_ids, logits_processor = logits_processor, stopping_criteria = stopping_criteria
)

tokenizer.batch_decode(outputs["sequences"], skip_special_tokens=True)


# inputs = tokenizer("Hello, I am Joe Biden", return_tensors="pt")
# tokens = model.generate(**inputs)
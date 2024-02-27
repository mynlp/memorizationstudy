from transformers import GPTNeoXForCausalLM, AutoTokenizer,  AutoModelForCausalLM,  LogitsProcessorList, MinLengthLogitsProcessor, StoppingCriteriaList,  MaxLengthCriteria
model = GPTNeoXForCausalLM.from_pretrained(
  "EleutherAI/pythia-160m-deduped",
  revision="step143000",
  cache_dir="./pythia-160m-deduped/step143000",
)

tokenizer = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-160m-deduped",
  revision="step143000",
  cache_dir="./pythia-160m-deduped/step143000",
)

model1 = GPTNeoXForCausalLM.from_pretrained(
  "EleutherAI/pythia-70m-deduped",
  revision="step143000",
  cache_dir="./pythia-70m-deduped/step143000",
)

tokenizer1 = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-70m-deduped",
  revision="step143000",
  cache_dir="./pythia-70m-deduped/step143000",
)

model2 = GPTNeoXForCausalLM.from_pretrained(
  "EleutherAI/pythia-410m-deduped",
  revision="step143000",
  cache_dir="./pythia-410m-deduped/step143000",
)

tokenizer2 = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-410m-deduped",
  revision="step143000",
  cache_dir="./pythia-410m-deduped/step143000",
)



#tokenizer = AutoTokenizer.from_pretrained("gpt2")
#model = AutoModelForCausalLM.from_pretrained("gpt2")

# set pad_token_id to eos_token_id because GPT2 does not have a PAD token
model.generation_config.pad_token_id = model.generation_config.eos_token_id
model.generation_config.output_hidden_states = True
model.generation_config.output_attentions = True
model.generation_config.output_scores = True
model.generation_config.return_dict_in_generate = True
model1.generation_config.pad_token_id = model.generation_config.eos_token_id
model1.generation_config.output_hidden_states = True
model1.generation_config.output_attentions = True
model1.generation_config.output_scores = True
model1.generation_config.return_dict_in_generate = True
model2.generation_config.pad_token_id = model.generation_config.eos_token_id
model2.generation_config.output_hidden_states = True
model2.generation_config.output_attentions = True
model2.generation_config.output_scores = True
model2.generation_config.return_dict_in_generate = True

model.eval()
model1.eval()
model2.eval()


inputs = tokenizer(["Hello, I am Joe Biden Mail", "Hello, I am Joe Biden Mail"], return_tensors="pt", truncation=True)
inputs1 = tokenizer1(["Hello, I am Joe Biden Mail", "Hello, I am Joe Biden Mail"], return_tensors="pt", truncation=True)
inputs2 = tokenizer2(["Hello, I am Joe Biden Mail", "Hello, I am Joe Biden Mail"], return_tensors="pt", truncation=True)

generations = model.generate(inputs["input_ids"], temperature = 0.0, top_k = 0, top_p = 0, max_length = 40, min_length = 40)
generations1 = model1.generate(inputs1["input_ids"], temperature = 0.0, top_k = 0, top_p = 0, max_length = 40, min_length = 40)
generations2 = model2.generate(inputs2["input_ids"], temperature = 0.0, top_k = 0, top_p = 0, max_length = 40, min_length = 40)


#hidden states of last word at the last layer
embedding = generations.hidden_states[-1][-1]
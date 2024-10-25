from transformers import GPTNeoXForCausalLM, AutoTokenizer,  AutoModelForCausalLM,  LogitsProcessorList, MinLengthLogitsProcessor, StoppingCriteriaList,  MaxLengthCriteria
from pythia.utils.mmap_dataset import MMapIndexedDataset
from transformers import GPTNeoXForCausalLM, AutoTokenizer
import torch
from utils import *


model_name = "EleutherAI/pythia-160m-deduped-v0"
CHECKPOINT= 143000
model = GPTNeoXForCausalLM.from_pretrained(
  model_name,
  revision=f"step{CHECKPOINT}",
  cache_dir=f"./pythia-160m-deduped/step{CHECKPOINT}",
)
model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(
  model_name,
  revision=f"step{CHECKPOINT}",
  cache_dir=f"./pythia-160m-deduped/step{CHECKPOINT}",
)
model.generation_config.pad_token_id = model.generation_config.eos_token_id
model.generation_config.output_hidden_states = True
model.generation_config.output_attentions = True
model.generation_config.output_scores = True
model.generation_config.return_dict_in_generate = True

prefix = 'undeduped_merge/document.bin'
if "deduped" in model_name:
    prefix = 'deduped_merge/document.bin'
print(prefix)
buff_size = 2049*1024*2
print("Building dataset")
mmap_ds = MMapIndexedDataset(prefix, skip_warmup=True)
context_tokens = []
true_continuation = []
i = 0
total_num_sequences = CHECKPOINT * 1024
NUM_PROCS = 1
RANK = 0
num_sequences_per_proc = total_num_sequences // NUM_PROCS
start_idx = num_sequences_per_proc * RANK
end_idx = num_sequences_per_proc * (RANK + 1) - 1
if RANK == (NUM_PROCS - 1):
    end_idx = total_num_sequences - 1
df1 = read_csv("/work/gk77/k77025/memorizationstudy/generate_results/memorization_evals_70m-deduped-v0_32_48_143000.csv")
df1 = df1[df1['0.0'] == 1]
listed = df1["0"].to_list()
batched_context_tokens = []
batched_true_continuation = []
for idx in listed[0:100]:
    data = mmap_ds[idx]
    context_tokens = data[:32].tolist()
    true_continuation = data[32:48].tolist()
    batched_context_tokens.append(context_tokens)
    batched_true_continuation.append(true_continuation)
i += len(context_tokens)
context_tokens = torch.tensor(batched_context_tokens).to('cuda')
true_continuation = torch.tensor(batched_true_continuation).to('cuda')
generations = model.generate(context_tokens, temperature = 0.0, top_k = 0, top_p = 0, max_length = 48, min_length = 48)
accuracies = (true_continuation == generations[0][:,32:48]).float().mean(axis=-1)
#generations 0  is the predicted tokenids (batch size, sequence length)
#generations 1 is the scores (continuation size, (batch_size, vocab size)) take the argmax of this to get the token ids of continuations
#generations 2 is the attentions states (continuation size, (num_layer, (batch_soze, num heads, context size, context size))))
#generations 3 is the hidden states (continuation size, (num_layer, (batch_size,size,context size,embeddings size))
#generations 4 is the past key value states (num_layer,(key value, (batch size, num heads, whole size, head dimension)))


# for i in range(len(context_tokens)):
#     print(f"Context:{tokenizer.batch_decode(context_tokens[i])}")
#     print(f"True Continuation:{tokenizer.batch_decode(true_continuation[i])}")
#     print(f"Generated Text:{tokenizer.batch_decode(generations[i])}")
#     print(f"Accuracy:{accuracies[i]}")
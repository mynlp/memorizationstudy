import torch

from transformers import AutoTokenizer, pipeline, StoppingCriteria, StoppingCriteriaList

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

model_name = "lambdalabs/pythia-70m-deduped-synthetic-instruct"
max_new_tokens = 1536
stop_token = "<|stop|>"


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_ids: list):
        self.keywords = keywords_ids

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        if input_ids[0][-1] in self.keywords:
            return True
        return False


tokenizer = AutoTokenizer.from_pretrained(
    model_name,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_tokens([stop_token])

stop_ids = [tokenizer.encode(w)[0] for w in [stop_token]]
stop_criteria = KeywordsStoppingCriteria(stop_ids)

generator = pipeline(
    "text-generation",
    model=model_name,
    device=device,
    max_new_tokens=max_new_tokens,
    torch_dtype=torch.float16,
    stopping_criteria=StoppingCriteriaList([stop_criteria]),
)

example = "How can I make an omelette."
text = "Question: {}\nAnswer:".format(example)

result = generator(
    text,
    num_return_sequences=1,
)

output = result[0]["generated_text"]

print(output)

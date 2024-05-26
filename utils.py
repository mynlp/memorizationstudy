import numpy as np
import pandas as pd
import torch
import pdb
import copy
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import GPTNeoXForCausalLM, AutoTokenizer,  AutoModelForCausalLM,  LogitsProcessorList, MinLengthLogitsProcessor, StoppingCriteriaList,  MaxLengthCriteria

class InstructDataset(Dataset):
    def __init__(self, json_list, tokenizer, PROMPT_DICT, ignore_index=-100):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.features = []

        for j in tqdm(json_list):
            # open_qaなど文脈情報が必要ない場合はinputカラムがないため、
            # inputカラムありなしでテンプレート文を分けている。
            if 'context' in j:
                source_text = PROMPT_DICT['prompt_input'].format_map(j)
            else:
                source_text = PROMPT_DICT['prompt_no_input'].format_map(j)

            # 指示文と回答文を結合し、文末にEOSトークンを挿入
            example_text = source_text + j['response'] + self.tokenizer.eos_token

            # 指示文のみ（「以下は、タスクを〜### 応答:」まで）をtokenize
            # ほしいのは指示文のlength
            source_tokenized = self.tokenizer(
                source_text,
                padding='longest',
                truncation=True,
                max_length=512,
                return_length=True,
                return_tensors='pt'
            )

            # 指示文と回答文を全てtokenize
            example_tokenized = self.tokenizer(
                example_text,
                padding='longest',
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )

            input_ids = example_tokenized['input_ids'][0]

            # LLMが生成してほしい正解の文章として入力文をそのままコピーする
            labels = copy.deepcopy(input_ids)

            # 指示文までの長さ
            source_len = source_tokenized['length'][0]

            # LLMに生成してほしい正解文章に指示文も含まれているので、
            # 指示文のところはCrossEntropyLossの損失を計算をしないようにIGNORE_INDEXとして-100で埋める
            labels[:source_len] = self.ignore_index

            self.features.append({
                'input_ids': input_ids,
                'labels': labels
            })

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


class InstructCollator():
    def __init__(self, tokenizer, ignore_index=-100):
        self.tokenizer = tokenizer
        self.ignore_index = -100

    def __call__(self, examples):
        input_batch = []
        label_batch = []
        for example in examples:
            input_batch.append(example['input_ids'])
            label_batch.append(example['labels'])

        input_ids = pad_sequence(
            input_batch, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        # labelsのpaddingトークンは先程と同様にignore_indexである-100で埋める
        labels = pad_sequence(
            label_batch, batch_first=True, padding_value=self.ignore_index
        )

        # attention_maskはbool値でもいいらしい
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }

def model_load(model_size, CHECKPOINT=143000):
    model = GPTNeoXForCausalLM.from_pretrained(
        f"EleutherAI/pythia-{model_size}-deduped-v0",
        revision=f"step{CHECKPOINT}",
        cache_dir=f"./pythia-{model_size}-deduped/step{CHECKPOINT}",
    ).eval()
    model = model.to_bettertransformer()
    model = model.cuda()
    return model

def read_csv(addr):
    csv_sheet = pd.read_csv(addr)
    return csv_sheet

def to_cpu(data):
    if isinstance(data, dict):
        return {k: to_cpu(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_cpu(v) for v in data]
    elif isinstance(data, tuple):
        return tuple(to_cpu(v) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.cpu()
    else:
        return data


def embedding_obtain(dataset, model, idx_list, context_size, continuation_size):
    batched_context_tokens = []
    batched_true_continuation = []
    for idx in idx_list:
        data = dataset[idx]
        context_tokens = data[:context_size].tolist()
        true_continuation = data[context_size:context_size+continuation_size].tolist()
        batched_context_tokens.append(context_tokens)
        batched_true_continuation.append(true_continuation)
    if torch.cuda.is_available():
        context_tokens = torch.tensor(batched_context_tokens).to('cuda')
        true_continuation = torch.tensor(batched_true_continuation).to('cuda')
    else:
        context_tokens = torch.tensor(batched_context_tokens)
        true_continuation = torch.tensor(batched_true_continuation)
    try:
        generations = model.generate(context_tokens, temperature=0.0, top_k=0, top_p=0, max_length=context_size+continuation_size, min_length=context_size+continuation_size)
        accuracies = (true_continuation == generations[0][:, context_size:context_size + continuation_size]).float().mean(axis=-1).cpu()
        generations = to_cpu(generations)
        return [generations, accuracies]
    except torch.cuda.OutOfMemoryError:
        generations = model.generate(context_tokens[:int(len(context_tokens)/2)], temperature=0.0, top_k=0, top_p=0, max_length=context_size+continuation_size, min_length=context_size+continuation_size)
        generations1 = model.generate(context_tokens[int(len(context_tokens)/2):], temperature=0.0, top_k=0, top_p=0, max_length=context_size+continuation_size, min_length=context_size+continuation_size)
        predictions=torch.concat((generations[0][:, context_size:context_size + continuation_size],
                      generations[1][:, context_size:context_size + continuation_size]), dim=0)
        accuracies = (true_continuation == predictions).float().mean(axis=-1)
        generations = to_cpu(generations)
        generations1 = to_cpu(generations1)
        results = []
        idx = 0
        for a, b in zip(generations.hidden_states, generations1.hidden_states):
            results.append([])
            for sub_a, sub_b in zip(a, b):
                results[idx].append(torch.cat((sub_a, sub_b), dim=0))
            idx += 1
        generations.hidden_states = results
        return [generations, accuracies]

def logits_obtain(dataset, model, idx_list, context_size, continuation_size):
    batched_context_tokens = []
    batched_true_continuation = []
    for idx in idx_list:
        data = dataset[idx]
        context_tokens = data[:context_size].tolist()
        true_continuation = data[context_size:context_size + continuation_size].tolist()
        batched_context_tokens.append(context_tokens)
        batched_true_continuation.append(true_continuation)
    context_tokens = torch.tensor(batched_context_tokens).to('cuda')
    true_continuation = torch.tensor(batched_true_continuation).to('cuda')
    batch_size = 10  # set batch size based on your GPU and model requirements
    highest_entropy_at_idx = []

    # process each batch
    for i in range(0, len(context_tokens), batch_size):
        batched_highest_entropy_at_idx = []
        batch_context_tokens = context_tokens[i:i + batch_size]
        batch_true_continuation = true_continuation[i:i + batch_size]

        # convert lists to tensors and move to GPU
        batched_context_tokens = torch.tensor(batch_context_tokens).to('cuda')
        batched_true_continuation = torch.tensor(batch_true_continuation).to('cuda')
        predicted_continuation = torch.zeros(batched_true_continuation.size()).long().to('cuda')
        for idx in range(1, context_size+continuation_size):
            if idx < context_size:
                model_outputs = model.module.generate(batch_context_tokens[:, :idx], temperature=0.0, top_k=0, top_p=0,
                                               max_length=idx + 1, min_length=idx + 1)
                logits = model_outputs["scores"]
                probability_scores = torch.nn.functional.softmax(logits[0], dim=1)
                entropy_scores = torch.distributions.Categorical(probs=probability_scores).entropy()
                batched_highest_entropy_at_idx.append(entropy_scores)
            else:
                temp_context_tokens = torch.cat((batch_context_tokens, predicted_continuation[:, :idx - context_size]), dim=1).cuda()
                model_outputs = model.module.generate(temp_context_tokens, temperature=0.0, top_k=0, top_p=0,
                                               max_length=idx + 1,
                                               min_length=idx + 1)
                predicted_continuation[:, idx-context_size] = model_outputs[0][:,-1].squeeze()
                logits = model_outputs["scores"]
                probability_scores = torch.nn.functional.softmax(logits[0], dim=1)
                entropy_scores = torch.distributions.Categorical(probs=probability_scores).entropy()
                batched_highest_entropy_at_idx.append(entropy_scores)
        # run model and get logits
        # model_outputs = model.generate(batched_context_tokens, temperature=0.0, top_k=0, top_p=0,
        #                                max_length=context_size + continuation_size,
        #                                min_length=context_size + continuation_size)
        # logits = model_outputs["scores"]
        # probability_scores = torch.nn.functional.softmax(logits[0], dim=1)
        # entropy_scores = torch.distributions.Categorical(probs=probability_scores).entropy()
        # # calculate entropy for each token in the context
        # for idx in range(continuation_size):
        #     probability_scores = torch.nn.functional.softmax(logits[idx], dim=1)
        #     entropy_scores = torch.distributions.Categorical(probs=probability_scores).entropy()
        #     batched_highest_entropy_at_idx.append(entropy_scores)

        batched_highest_entropy_at_idx = torch.stack(batched_highest_entropy_at_idx)#100,16
        highest_entropy_at_idx.append(batched_highest_entropy_at_idx)

    # convert list of tensors into a single tensor
    highest_entropy_at_idx = torch.concat(highest_entropy_at_idx,dim=1).cpu()
    # model_outputs = model.generate(context_tokens, temperature=0.0, top_k=0, top_p=0, max_length=context_size+continuation_size, min_length=context_size+continuation_size)
    # logits = model_outputs["scores"]
    # highest_entropy_at_idx = []
    # for idx in range(continuation_size):
    #     probability_scores = torch.nn.functional.softmax(logits[idx], dim=1)
    #     entropy_scores = torch.distributions.Categorical(probs=probability_scores).entropy()
    #     #pdb.set_trace()
    #     highest_entropy_at_idx.append(entropy_scores)
    return highest_entropy_at_idx
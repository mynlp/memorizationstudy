import os
import logging
import time
import datetime
import torch
import torch.distributed as dist
import transformers.utils as transformer_utils
import torch.multiprocessing as mp
from pythia.utils.mmap_dataset import MMapIndexedDataset
from transformers import GPTNeoXForCausalLM
import argparse
from utils import *
import pdb


def generate_dataset(model, batch_size, context_size, continuation_size, start_seq_idx, end_seq_idx, mp_queue, prefetch_max=128):
    prefix = 'undeduped_merge/document.bin'
    if "deduped" in model:
        prefix = 'deduped_merge/document.bin'
    print(prefix)
    buff_size = 2049*batch_size*2
    print("Building dataset")
    mmap_ds = MMapIndexedDataset(prefix, skip_warmup=True)
    context_tokens = []
    true_continuation = []
    i = 0
    for i in range(start_seq_idx, end_seq_idx + 1, batch_size):
        data = mmap_ds[i:i + batch_size]
        context_tokens.extend(data[:, :context_size].tolist())
        true_continuation.extend(data[:, context_size:context_size+continuation_size].tolist())
        i += len(context_tokens)

        if len(context_tokens) == batch_size:
            # (start index of batch, context tokens, true continuation)
            mp_queue.put((
                i - len(context_tokens),
                context_tokens, true_continuation))
            context_tokens = []
            true_continuation = []
            while mp_queue.qsize() > prefetch_max:
                time.sleep(0.05)
    if len(context_tokens) > 0:
        mp_queue.put((i - len(context_tokens) + 1, context_tokens, true_continuation))
        context_tokens = []
        true_continuation = []

    mp_queue.put((None, None, None))


def score(model, context_tokens, true_continuation, context_size, continuation_size):
    """Calculate memorization score from context tokens and true continuation

    Performs greedy generation from context tokens and calculates memorization score

    Args:
        model (transformers.GPTNeoXForCausalLM): Pythia model instance being evaluated
        context_tokens (torch.Tensor): Context token indicies of shape (batch_size, 32)
        true_continuation (torch.Tensor): True continuation indicies of shape (batch_size, 32)

    Returns:
        accuracies (torch.Tensor): Accuracies of shape (batch_size,)
    """
    with torch.no_grad():
        context_tokens = torch.tensor(context_tokens).to('cuda')
        true_continuation = torch.tensor(true_continuation).to('cuda')
        if isinstance(model, torch.nn.DataParallel):
            generations = model.module.generate(context_tokens, temperature = 0.0, top_k = 0, top_p = 0, max_length = context_size+continuation_size, min_length = context_size+continuation_size)
        else:
            generations = model.generate(context_tokens, temperature = 0.0, top_k = 0, top_p = 0, max_length = context_size+continuation_size, min_length = context_size+continuation_size)
        accuracies = (true_continuation == generations[:,context_size:context_size+continuation_size]).float().mean(axis=-1)
        return accuracies.cpu()

def inference(model, model_name, checkpoint, batch_size, context_size, continuation_size, mp_queue):
    #dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print(model)
    # if f"memorization_evals_{args.model}_{args.context_size}_{args.context_size+args.continuation_size}_{args.checkpoint}.csv" in os.listdir("generate_results"):
    #     df = pd.read_csv(f"generate_results/memorization_evals_{args.model}_{args.context_size}_{args.context_size+args.continuation_size}_{args.checkpoint}.csv", index_col=0)
    #     start_idx = len(df)
    # else:

    # Dataset Initialization
    model = model.half().to("cuda")
    model = torch.nn.DataParallel(model)
    model.eval()
    print("Loaded Model")
    all_memorization_evals = []
    all_memorization_evals_values = []
    memorization_evals = []
    memorization_evals_values = []
    iters = 0
    debug_count = 0
    while (True):
        try:
            t = time.time()
            idx, context, true_continuation = mp_queue.get()
            if idx is None:
                mp_queue.close()
                break

            idx = idx
            print(f"Loading data took {time.time() - t:.3}s")
            t = time.time()
            accuracies = score(model, context, true_continuation, context_size, continuation_size)
            for acc in accuracies:
                all_memorization_evals.append(f'{idx},{acc}')
                all_memorization_evals_values.append([idx, acc.tolist()])
                memorization_evals.append(f'{idx},{acc}')
                memorization_evals_values.append([idx, acc.tolist()])
                idx += 1
                debug_count += 1
            print(f"Generation until {idx} took {time.time() - t:.3}s")
            del idx, context, true_continuation
            iters += 1
            # if (idx / 1024) % 1430 == 0:
            #     print(f"Processed {iters} iterations until {idx}")
            #     if f"memorization_evals_{args.model}_{args.context_size}_{args.context_size + args.continuation_size}_{args.checkpoint}.csv" in os.listdir(
            #             "generate_results"):
            #         df = pd.read_csv(
            #             f"generate_results/memorization_evals_{args.model}_{args.context_size}_{args.context_size + args.continuation_size}_{args.checkpoint}.csv",
            #             index_col=0)
            #         cache = pd.DataFrame(memorization_evals_values, columns=["idx", "score"])
            #         df = pd.concat([df, cache]).reset_index(drop=True)
            #         df.to_csv(
            #             f"generate_results/memorization_evals_{args.model}_{args.context_size}_{args.context_size + args.continuation_size}_{args.checkpoint}.csv")
            #         print("Saved Merged Results")
            #     else:
            #         df = pd.DataFrame(memorization_evals_values, columns=["idx", "score"])
            #         df.to_csv(
            #             f"generate_results/memorization_evals_{args.model}_{args.context_size}_{args.context_size + args.continuation_size}_{args.checkpoint}.csv")
            #         print("Saved Merged Results")
            #     memorization_evals = []
            #     memorization_evals_values = []
        except StopIteration:
            print("Break")
            break
    df = pd.DataFrame(all_memorization_evals_values, columns=["idx", "score"])
    df.to_csv(
        f"generate_results/memorization_evals_{model}_{context_size}_{context_size + continuation_size}_{checkpoint}.csv")
    #ds_process.join()

def main():
    batch_size = 4096
    context_size = 48
    continuation_size = 16
    model_name = "70m-deduped-v0"
    checkpoint = 143000
    world_size = 8
    print("start")
    model = GPTNeoXForCausalLM.from_pretrained(
        f"EleutherAI/pythia-{model_name}",
        revision=f'step{checkpoint}',
    )
    mp_queue = mp.Queue()
    total_num_sequences = checkpoint * batch_size
    num_sequences_per_proc = total_num_sequences // world_size
    start_idx = 0
    end_idx = total_num_sequences - 1
    ds_process = mp.Process(target=generate_dataset, args=(model_name, batch_size, context_size, continuation_size, start_idx, end_idx, mp_queue))
    ds_process.start()
    inference(model, model_name, checkpoint, batch_size, context_size, continuation_size, mp_queue)
    ds_process.join()
    # for rank in range(world_size):
    #     p = mp.Process(target=inference, args=(rank, model, model_name, checkpoint, batch_size, context_size, continuation_size, world_size))
    #     start_idx = num_sequences_per_proc * rank
    #     end_idx = num_sequences_per_proc * (rank + 1) - 1
    #     if rank == (world_size - 1):
    #         end_idx = total_num_sequences - 1
    #     ds_process = mp.Process(target=generate_dataset, args=(
    #     model_name, batch_size, context_size, continuation_size, start_idx, end_idx, mp_queue))
    #     ds_process.start()


    # # Model initialization
    # transformer_utils.logging.set_verbosity_error()
    # # Calculate start and end sequence indicies
    # total_num_sequences = args.checkpoint * args.batch_size
    # num_sequences_per_proc = total_num_sequences // NUM_PROCS
    # start_idx = num_sequences_per_proc * RANK
    #
    # end_idx = num_sequences_per_proc * (RANK + 1) - 1
    # if RANK == (NUM_PROCS - 1):
    #     end_idx = total_num_sequences - 1
    #
    # # Dataset Initialization
    # mp_queue = mp.Queue()
    # ds_process = mp.Process(target=generate_dataset, args=(args.model, args.batch_size, args.context_size, args.continuation_size, start_idx, end_idx, mp_queue))
    # ds_process.start()

    # Model initialization

if __name__ == '__main__':
    main()
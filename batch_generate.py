import os
import logging
import time
import datetime
import torch
import torch.distributed as dist
import transformers.utils as transformer_utils
import multiprocessing as mp
from pythia.utils.mmap_dataset import MMapIndexedDataset
from transformers import GPTNeoXForCausalLM, AutoModelForCausalLM
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

def main():
    paser = argparse.ArgumentParser()
    paser.add_argument("--batch_size", type=int, default=256)
    paser.add_argument("--context_size", type=int, default=32)
    paser.add_argument("--continuation_size", type=int, default=16)
    paser.add_argument("--model", type=str, default="6.9b-deduped-v0")
    paser.add_argument("--checkpoint", type=int, default=143000)
    paser.add_argument("--rank", type=int, default=0)
    args = paser.parse_args()
    #BATCH_SIZE = 1024
    #LOG_INTERVAL = 100
    RANK = args.rank#int(os.environ['RANK'])
    NUM_PROCS = 64
    #os.environ['MODEL'] = MODEL
   #int(os.environ['CHECKPOINT'])
    #os.environ['CHECKPOINT'] = str(CHECKPOINT)
    #os.environ['MASTER_ADDR'] = "127.0.0.1"
    #os.environ['MASTER_PORT'] = '13443'
    #logging.basicConfig(format = f'rank-{RANK}:' + '%(levelname)s:%(message)s', level = print)
    print(f"Initializing torch distributed with gpus {torch.cuda.device_count()}")
    #torch.cuda.set_device(RANK)
    #dist.init_process_group(
    #     "nccl",
    #     world_size=NUM_PROCS,
    #     rank=RANK
    #)
    #store = dist.TCPStore(os.environ['MASTER_ADDR'], port=13443,
    #                       world_size=NUM_PROCS, is_master=RANK == 0, timeout=datetime.timedelta(hours=3))
    print("start")

    #dist.barrier()

    # Model initialization
    transformer_utils.logging.set_verbosity_error()

    # Calculate start and end sequence indicies
    total_num_sequences = args.checkpoint * 1024
    num_sequences_per_proc = total_num_sequences // NUM_PROCS
    start_idx = num_sequences_per_proc * RANK
    if f"memorization_evals_{args.model}_{args.context_size}_{args.context_size+args.continuation_size}_{args.checkpoint}_{RANK}.csv" in os.listdir("generate_results"):
         exsit_df = pd.read_csv(f"generate_results/memorization_evals_{args.model}_{args.context_size}_{args.context_size+args.continuation_size}_{args.checkpoint}_{RANK}.csv",
                          index_col=0)
         start_idx = len(exsit_df)+start_idx
         file_exsits = True
    else:
         file_exsits = False
    end_idx = num_sequences_per_proc * (RANK + 1) - 1
    if RANK == (NUM_PROCS - 1):
        end_idx = total_num_sequences - 1
    print(f"Start from idx {start_idx}")
    print(f"End at idx {end_idx}")
    # Dataset Initialization
    mp_queue = mp.Queue()
    ds_process = mp.Process(target=generate_dataset, args=(args.model, args.batch_size, args.context_size, args.continuation_size, start_idx, end_idx, mp_queue))
    ds_process.start()

    # Model initialization
    #model = AutoModelForCausalLM.from_pretrained(f"EleutherAI/pythia-{args.model}", revision=f'step{args.checkpoint}', load_in_8bit=True, device_map="cuda:0")
    model = GPTNeoXForCausalLM.from_pretrained(
        f"EleutherAI/pythia-{args.model}",
        revision=f'step{args.checkpoint}',
    ).half().eval()
    model = model.to_bettertransformer()
    if torch.cuda.device_count() > 1:
        print(f"use {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model,device_ids=[0, 1])
    else:
        model = model.cuda()
    #dist.barrier()
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
            accuracies = score(model, context, true_continuation, args.context_size, args.continuation_size)

            for acc in accuracies:
                all_memorization_evals.append(f'{idx},{acc}')
                all_memorization_evals_values.append([idx, acc.tolist()])
                memorization_evals.append(f'{idx},{acc}')
                memorization_evals_values.append([idx, acc.tolist()])
                idx += 1
                debug_count += 1
            print(f"Generation until {idx} took {time.time() - t:.3}s")
            #dist.barrier()
            iters += 1
            if iters % 500 == 0:
                print(f"Processed {iters} iterations until {idx}")
                if file_exsits:
                    new_df = pd.DataFrame(memorization_evals_values, columns=["idx", "score"])
                    df = pd.concat([exsit_df, new_df]).reset_index(drop=True)
                    df.to_csv(f"generate_results/memorization_evals_{args.model}_{args.context_size}_{args.context_size+args.continuation_size}_{args.checkpoint}_{RANK}.csv")
                    print("Saved Merged Results")
                else:
                    df = pd.DataFrame(memorization_evals_values, columns=["idx", "score"])
                    df.to_csv(f"generate_results/memorization_evals_{args.model}_{args.context_size}_{args.context_size+args.continuation_size}_{args.checkpoint}_{RANK}.csv")
                    print("Saved Merged Results")
        except StopIteration:
            print("Break")
            break
    if file_exsits:
        new_df = pd.DataFrame(memorization_evals_values, columns=["idx", "score"])
        df = pd.concat([exsit_df, new_df]).reset_index(drop=True)
        df.to_csv(
            f"generate_results/memorization_evals_{args.model}_{args.context_size}_{args.context_size + args.continuation_size}_{args.checkpoint}_{RANK}.csv")
    else:
        df = pd.DataFrame(all_memorization_evals_values, columns=["idx", "score"])
        df.to_csv(
            f"generate_results/memorization_evals_{args.model}_{args.context_size}_{args.context_size + args.continuation_size}_{args.checkpoint}_{RANK}.csv")
    ds_process.join()
    # dist.barrier()

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
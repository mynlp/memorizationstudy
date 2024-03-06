import os
import logging
import time
import datetime
import torch
import torch.distributed as dist
import transformers.utils as transformer_utils
import multiprocessing as mp
from pythia.utils.mmap_dataset import MMapIndexedDataset
from transformers import GPTNeoXForCausalLM
import argparse
from utils import *
import pdb
def generate_dataset(args, start_seq_idx, end_seq_idx, mp_queue, prefetch_max=128):
    prefix = 'undeduped_merge/document.bin'
    if "deduped" in args.model:
        prefix = 'deduped_merge/document.bin'
    print(prefix)
    buff_size = 2049*args.batch_size*2
    print("Building dataset")
    mmap_ds = MMapIndexedDataset(prefix, skip_warmup=True)
    context_tokens = []
    true_continuation = []
    i = 0
    for i in range(start_seq_idx, end_seq_idx + 1, args.batch_size):
        data = mmap_ds[i:i + args.batch_size]
        context_tokens.extend(data[:, :args.context_size].tolist())
        true_continuation.extend(data[:, args.context_size:args.context_size+args.continuation_size].tolist())
        i += len(context_tokens)

        if len(context_tokens) == args.batch_size:
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
        generations = model.generate(context_tokens, temperature = 0.0, top_k = 0, top_p = 0, max_length = context_size+continuation_size, min_length = context_size+continuation_size)
        accuracies = (true_continuation == generations[:,context_size:context_size+continuation_size]).float().mean(axis=-1)
        return accuracies.cpu()


def main():
    paser = argparse.ArgumentParser()
    paser.add_argument("--batch_size", type=int, default=1024)
    paser.add_argument("--context_size", type=int, default=48)
    paser.add_argument("--continuation_size", type=int, default=16)
    paser.add_argument("--model", type=str, default="70m-deduped-v0")
    paser.add_argument("--checkpoint", type=int, default=143000)
    args = paser.parse_args()
    RANK = int(os.environ['RANK'])
    NUM_PROCS = int(os.environ['WORLD_SIZE'])
    logging.basicConfig(format = f'rank-{RANK}:' + '%(levelname)s:%(message)s', level = logging.INFO)
    logging.info(f"Initializing torch distributed with gpus {torch.cuda.device_count()}")
    print("start")
    torch.cuda.set_device(RANK)

    dist.init_process_group(
        "nccl",
        world_size=NUM_PROCS,
        rank=RANK
    )
    store = dist.TCPStore(os.environ['MASTER_ADDR'], port=12125,
                          world_size=NUM_PROCS, is_master=RANK == 0, timeout=datetime.timedelta(hours=3))

    dist.barrier()
    transformer_utils.logging.set_verbosity_error()

    # Calculate start and end sequence indicies
    total_num_sequences = args.checkpoint * args.batch_size
    num_sequences_per_proc = total_num_sequences // NUM_PROCS
    if f"memorization_evals_{args.model}_{args.context_size}_{args.context_size + args.continuation_size}_{args.checkpoint}_{RANK}.csv" in os.listdir(
            "generate_results"):
        df = pd.read_csv(
            f"generate_results/memorization_evals_{args.model}_{args.context_size}_{args.context_size + args.continuation_size}_{args.checkpoint}_{RANK}.csv",
            index_col=0)
        start_idx = len(df)
    else:
        start_idx = num_sequences_per_proc * RANK
    end_idx = num_sequences_per_proc * (RANK + 1) - 1
    if RANK == (NUM_PROCS - 1):
        end_idx = total_num_sequences - 1

    # Dataset Initialization
    mp_queue = mp.Queue()
    ds_process = mp.Process(target=generate_dataset, args=(args, start_idx, end_idx, mp_queue))
    ds_process.start()

    # Model initialization
    model = GPTNeoXForCausalLM.from_pretrained(
        f"EleutherAI/pythia-{args.model}",
        use_cache=False,
        revision=f'step{args.checkpoint}',
    ).half().eval().cuda()

    dist.barrier()
    logging.info("Loaded Model")

    # Run generations
    memorization_evals = []
    memorization_evals_values = []
    iters = 0
    while (True):
        try:
            t = time.time()
            idx, context, true_continuation = mp_queue.get()
            if idx is None:
                mp_queue.close()
                break

            idx = idx
            logging.info(f"Loading data took {time.time() - t:.3}s")
            t = time.time()
            accuracies = score(model, context, true_continuation, args.context_size, args.continuation_size)

            for acc in accuracies:
                memorization_evals.append(f'{idx},{acc}')
                memorization_evals_values.append([idx, acc.tolist()])
                idx += 1

            logging.info(f"Generation uptil {idx} took {time.time() - t:.3}s")
            dist.barrier()
            iters += 1
            if iters % 500 == 0:
                df = pd.DataFrame(memorization_evals_values, columns=["idx", "score"])
                df.to_csv(f"generate_results/memorization_evals_{args.model}_{args.context_size}_{args.context_size + args.continuation_size}_{args.checkpoint}_{RANK}.csv")
        except StopIteration:
            break

    ds_process.join()
    dist.barrier()
    df = pd.DataFrame(memorization_evals_values, columns=["idx", "score"])
    df.to_csv(f"generate_results/memorization_evals_{args.model}_{args.context_size}_{args.context_size + args.continuation_size}_{args.checkpoint}_{RANK}.csv")
    with open(f"experiment_cache/memorization_evals_{args.model}_{args.context_size}_{args.context_size + args.continuation_size}_{args.checkpoint}.txt", "w") as f:
        f.write(f"{RANK} done\n")

if __name__ == '__main__':
    main()
import pandas as pd
from argparse import ArgumentParser
import os

args = ArgumentParser()
args.add_argument("--model_name", type=str, default="1b-deduped-v0")
args.add_argument("--checkpoint", type=int, default=143000)
args.add_argument("--rank_size", type=int, default=64)
args.add_argument("--context_size", type=int, default=32)
args.add_argument("--continuation_size", type=int, default=16)
args = args.parse_args()

check_dict = {}
for rank in range(args.rank_size):
    check_dict[rank] = False
# with open(f"experiment_cache/memorization_evals_{args.model_name}_{args.context_size}_{args.context_size + args.continuation_size}_{args.checkpoint}.txt", "r") as f:
#     for line in f:
#         rank_idx, _ = line.split()
#         check_dict[int(rank_idx)] = True
#if all(check_dict.values()):
result = []
for rank in range(args.rank_size):
    print(f"Loading csv file idx {rank}...")
    file = f"generate_results/memorization_evals_{args.model_name}_{args.context_size}_{args.context_size + args.continuation_size}_{args.checkpoint}_{rank}.csv"
    df = pd.read_csv(file, index_col=0)
    result.append(df)
result = pd.concat(result, ignore_index=True)
result.to_csv(f"generate_results/memorization_evals_{args.model_name}_{args.context_size}_{args.context_size + args.continuation_size}_{args.checkpoint}.csv")
    #os.system(f"rm generate_results/memorization_evals_{args.model_name}_{args.context_size}_{args.context_size + args.continuation_size}_{args.checkpoint}_*.csv")


#generate_results/memorization_evals_1b-deduped-v0_32_48_143000.csv
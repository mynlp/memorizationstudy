import pandas as pd
from argparse import ArgumentParser
import os

args = ArgumentParser()
args.add_argument("--model_name", type=str, default="70m-deduped-v0")
args.add_argument("--checkpoint", type=int, default=143000)
args.add_argument("--rank_size", type=int, default=8)
args.add_argument("--context_size", type=int, default=96)
args.add_argument("--continuation_size", type=int, default=16)
args = args.parse_args()

result = []
for rank in range(args.rank_size):
    file = f"generate_results/memorization_evals_{args.model_name}_{args.context_size}_{args.context_size + args.continuation_size}_{args.checkpoint}_{rank}.csv"
    df = pd.read_csv(file, index_col=0)
    result.append(df)
result = pd.concat(result, ignore_index=True)
result.to_csv(f"generate_results/memorization_evals_{args.model_name}_{args.context_size}_{args.context_size + args.continuation_size}_{args.checkpoint}.csv")
os.system("rm generate_results/memorization_evals_70m-deduped-v0_96_112_143000_*.csv")



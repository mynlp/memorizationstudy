import pandas as pd
import os


for file in os.listdir("generate_results"):
    if file in ["memorization_evals_70m-deduped-v0_32_48_143000.csv", "memorization_evals_70m-deduped-v0_32_64_143000.csv",
                "memorization_evals_70m-deduped-v0_32_80_143000.csv", "memorization_evals_70m-deduped-v0_32_96_143000.csv",
                "memorization_evals_70m-deduped-v0_32_128_143000.csv", "memorization_evals_160m-deduped-v0_32_48_143000.csv",
                "memorization_evals_160m-deduped-v0_32_64_143000.csv", "memorization_evals_410m-deduped-v0_32_48_143000.csv",
                ]:
            df = pd.read_csv(file, names=["idx", "score"])

    elif file in ["memorization_evals_160m-deduped-v0_32_96_143000.csv", "memorization_evals_160m-deduped-v0_32_128_143000.csv",
                  "memorization_evals_410m-deduped-v0_32_64_143000.csv", "memorization_evals_410m-deduped-v0_32_128_143000.csv",
                  "memorization_evals_1b-deduped-v0_32_48_143000.csv", "memorization_evals_1b-deduped-v0_32_64_143000.csv", "memorization_evals_1b-deduped-v0_32_96_143000.csv",
                  "memorization_evals_1b-deduped-v0_32_128_143000.csv"]:
        df = pd.read_csv(file, index_col=0)
        df = df.rename(columns={"0":"idx", "0.0":"score"})





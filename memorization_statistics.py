import pandas as pd
from matplotlib import pyplot as plt


dataset_names = ["wikipedia_(en)", "pile_cc", "arxiv", "dm_mathematics", "github", "hackernews", "pubmed_central", "full_pile"]
model_size = "410m"
for dataset_name in dataset_names:
    table = pd.read_csv(f"mem_score_online/{model_size}/{dataset_name}_mem_score.csv", index_col=0)
    member_table= table[table["set_name"] == "member"]
    nonmember_table = table[table["set_name"] == "nonmember"]
    member_scores = member_table["mem_score"].values
    nonmember_scores = nonmember_table["mem_score"].values
    plt.hist(member_scores, bins=100, alpha=0.5, label='member')
    plt.hist(nonmember_scores, bins=100, alpha=0.5, label='nonmember')
    plt.legend(loc='upper right')
    plt.title(f"{dataset_name} Mem Score Distribution")
    plt.savefig(f"mem_score_online/{model_size}/{dataset_name}_mem_score.png")
    plt.show()

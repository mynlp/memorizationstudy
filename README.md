# Memorization Study
## Directory Explanation
1. generate_results: This contains the directory for the sentence idx and the memorization scores of this sentence.
The name of the file follow the format of memorization_evals_{model_size}_deduped-v0\_\{context size}\_\{continuation size}\_143000.csv which has two columns idx and scores.
2. dedup_data: This contains the original deduplicated data.
3. dedup_merge: This contains the merged deduplicated data.
4. undeduped_data: This contains the original undeduplicated data.
5. undedup_merge: This contains the merged undeduplicated data.
6. pythia: means the pythia package
## File expalanation
1. run_generate.sh: This initiatiaste the batch_generate.py script. The input parameters are model size, checkpoint (usually the last step), batch size (usually fixed), context size and continuation size.
2. data_download.py: Used to download the pre-train data. possibly do not have to use it again.
3. cluster.py: Sample different memorized/unmemorized data points and apply dimension reduction and show in a figure.
4. clmtraing.py: Trains a model on causal language modelling task.
5. embedding_obtain,py: A script shows how to obtain hiddent state embedding for Pythia or any other model.
6. generate.py, csv_process.py, csv_reformat.py are just some helper scripts or format conversion scripts may not be used again.
7. example_explore.py: A script to show to make a single example generation.
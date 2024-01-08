from huggingface_hub import hf_hub_download, snapshot_download
snapshot_download(repo_id="EleutherAI/pile-standard-pythia-preshuffled", repo_type="dataset", cache_dir="deduped_data")
snapshot_download(repo_id="EleutherAI/pile-deduped-pythia-preshuffled", repo_type="dataset", cache_dir="undeduped_data")
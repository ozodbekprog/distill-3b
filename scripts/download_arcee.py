import os
from huggingface_hub import snapshot_download

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dest = os.path.join(root, "data", "llama405b-logits")

path = snapshot_download(
    "arcee-ai/LLama-405B-Logits",
    repo_type="dataset",
    local_dir=dest,
    max_workers=4,
)
print("DONE", path)

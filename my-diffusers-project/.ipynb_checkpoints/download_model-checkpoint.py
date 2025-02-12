from huggingface_hub import snapshot_download

model_id = "stabilityai/stable-diffusion-3.5-medium"  # Replace with the model you want
#or model_id = "stabilityai/stable-diffusion-xl-base-1.0"
# Optional: specify a cache directory (defaults to ~/.cache/huggingface/hub)
cache_dir = "/root/.cache/huggingface/hub"

snapshot_download(repo_id=model_id, cache_dir=cache_dir, local_files_only=False,  repo_type="model")

print(f"Model {model_id} downloaded to {cache_dir}")
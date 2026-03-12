from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="dragonkue/colbert-ko-0.1b",
    local_dir="src/retrievers/models/dragonkue/colbert-ko-0.1b"
)
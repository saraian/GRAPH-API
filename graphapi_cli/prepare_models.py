"""Executed inside Docker: acquire missing existing inference assets, keep supplied files."""
from pathlib import Path
import shutil
import sys


def main():
    from huggingface_hub import hf_hub_download, snapshot_download
    destination = Path(sys.argv[1])
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("l2_encoder.onnx", "l2_decoder.onnx"):
        output = destination / name
        if not output.is_file() or output.stat().st_size == 0:
            cached = hf_hub_download("mit-han-lab/efficientvit-sam", "onnx/" + name)
            shutil.copyfile(cached, output)
        print(f"ready: {output}", flush=True)
    for model in ("sentence-transformers/all-MiniLM-L6-v2", "openai/clip-vit-base-patch32", "intfloat/e5-small-v2"):
        snapshot_download(model)
        snapshot_download(model, local_files_only=True)
        print(f"cached: {model}", flush=True)


if __name__ == "__main__":
    main()

"""Record the current machine instead of copying the reference-run environment."""
import importlib.metadata
import platform
from kkt_agent.io import dump_json

info = {"python": platform.python_version(), "platform": platform.platform()}
for package in ["torch", "transformers", "trl", "peft", "numpy", "scipy", "osqp",
                "datasets", "accelerate", "pandas", "matplotlib"]:
    try:
        info[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        info[package] = None
try:
    import torch
    info.update(cuda=torch.cuda.is_available(), cuda_version=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
except ImportError:
    info.update(cuda=False, cuda_version=None, gpu=None)
dump_json("results/environment.json", info)
print(info)

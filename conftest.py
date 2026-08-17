import os
from pathlib import Path

# Ensure CUDA headers are found by CuPy in all pytest workers
if "CUDA_PATH" not in os.environ and Path("/usr/include/cuda.h").exists():
    os.environ["CUDA_PATH"] = "/usr"

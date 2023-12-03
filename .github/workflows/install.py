import subprocess

subprocess.run(["sudo", "apt", "update", "-y"])
subprocess.run(["sudo", "apt", "install", "libssl-dev", "-y"])
subprocess.run(["sudo", "apt", "install", "pkg-config", "-y"])

try:
    import torch
    works = True
except:
    works = False

if works:
    first = subprocess.run(["sudo", "find", "/", "-name", "libtorch_cpu.so"]).stdout.split("\n")
else:
    first = []

nvcc_release = subprocess.run(["nvcc", "--version"])
assert nvcc_release.returncode == 0

nvcc_release = nvcc_release.stdout.split("\n")[3] #Cuda compilation tools, release 11.5, V11.5.119
nvcc_release = nvcc_release.split("release ")[1] #['Cuda compilation tools, ', '11.5, V11.5.119']
nvcc_release = float(nvcc_release[1].split(","))

print(f"Got nvcc version {nvcc_release}")
if nvcc_release<=11.8:
    subprocess.run(["pip", "install", "torch==2.1.0", "torchvision==0.16.0", "torchaudio==2.1.0", "--index-url", "https://download.pytorch.org/whl/cu118"])
else:
    subprocess.run(["pip", "install", "torch==2.1.0", "torchvision==0.16.0", "torchaudio==2.1.0", "--index-url", "https://download.pytorch.org/whl/cu121"])

after = subprocess.run(["sudo", "find", "/", "-name", "libtorch_cpu.so"]).stdout.split("\n")
different = list(filter(lambda x: x not in first, after))[0]

with open("~/.bashrc", "a") as f:
    f.write("# candle-vllm")
    f.write(f"export LD_LIBRARY_PATH={different}:$LD_LIBRARY_PATH")
    f.write("export LIBTORCH_USE_PYTORCH=1")

subprocess.run(["source", "~/.bashrc"])
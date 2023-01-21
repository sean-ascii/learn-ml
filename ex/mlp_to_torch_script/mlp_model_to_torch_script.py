import torch
from torch import nn

from pathlib import Path

# same with mlp/mlp_pytorch.py
model = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))

model_file_path = str(Path(__file__).resolve().parents[2]) + "/data/model/mlp_mnist_classfity.pth"
model.load_state_dict(torch.load(model_file_path, map_location="cpu"))

sample = torch.rand(1, 1, 28, 28)
trace_model = torch.jit.trace(model, sample)
jit_file_path = str(Path(__file__).resolve().parents[2]) + "/data/model/mlp_mnist_classfity.jit"
trace_model.save(jit_file_path)

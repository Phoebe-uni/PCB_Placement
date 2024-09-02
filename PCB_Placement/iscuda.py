import torch

# 检查CUDA是否可用
cuda_available = torch.cuda.is_available()

if cuda_available:
    print("CUDA is available.")
else:
    print("CUDA is not available.")
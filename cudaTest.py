import torch

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")  # Use GPU
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")  # Use CPU
    print("CUDA is not available. Using CPU.")

# Create a tensor and move it to the GPU
x = torch.randn(3, 3).to(device)
print(x)
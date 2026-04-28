import torch

print(f'Pytorch version: {torch.__version__}')

print(f"mps available: {torch.mps.is_available()}")
if torch.cuda.is_available():
    print(f'gpu: {torch.cuda.get_device_name(0)}')
else:
    print("running in cpu mode")

a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
print(torch.matmul(a, b))
print(a @ b)
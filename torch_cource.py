import torch

a = torch.tensor([1, 2, 3])

v = torch.tensor([10, 20, 30, 40, 50, 60, 70])

print(v[0])
print(v[-1])

print(v[1:4])
print(v[:3])
print(v[4:])
print(v[-3:])

print(v[::2])
print(v[::-1])


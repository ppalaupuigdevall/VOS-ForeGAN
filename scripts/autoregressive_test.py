import torch
import torch.nn as nn


a = torch.rand(3,3)

# This will be share by both iterations and will make the second backward fail !
b = a * a

for i in range(10):
    d = torch.mean(b * b)
    # The first here will work but the second will not !
    d.backward()
    print("back")
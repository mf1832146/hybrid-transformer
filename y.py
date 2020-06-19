import torch


if __name__ == '__main__':
    a = torch.rand([1,2,3])
    print(a[:, -1])
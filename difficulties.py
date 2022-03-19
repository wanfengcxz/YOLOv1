import torch


def three_dots():
    x = torch.arange(24)
    x = torch.reshape(x, (2, 3, 4))
    print("x: ", x)
    y1 = x[..., 0:1]
    print('x[..., 0:1]:', y1, y1.shape)
    y2 = x[..., 0]
    print('x[..., 0]:', y2, y2.shape)


def torch_max():
    x1 = torch.rand((2, 3, 4))
    x2 = torch.rand((2, 3, 4))
    print('x1:', x1)
    print('x2:', x2)
    output = torch.max(x1, x2)
    print('torch.max(x1, x2):', output)


if __name__ == '__main__':
    torch_max()

import torch


def three_dots():
    x = torch.arange(24)
    x = torch.reshape(x, (2, 3, 4))
    print("x: ", x)
    y1 = x[..., 0:1]
    print('x[..., 0:1]:', y1)
    print(y1.shape)
    y2 = x[..., 0]
    print('x[..., 0]:', y2)
    print(y2.shape)
    x = torch.tensor([1,2,3])
    print(x[..., 0:1])


def torch_max():
    x1 = torch.rand((2, 3, 4))
    x2 = torch.rand((2, 3, 4))
    print('x1:', x1)
    print('x2:', x2)
    output = torch.max(x1, x2)
    print('torch.max(x1, x2):', output)


def torch_max_dim():
    # x1.shape: (B, batch_size, S, S, 1)
    x1 = torch.rand((2, 3, 7, 7, 1))
    # print('x1:', x1)
    max_values, indices = torch.max(x1, dim=0)
    print('max_values shape: ', max_values.shape)
    print('indices shape: ', indices.shape)


def tensor_unsqueeze():
    x = torch.zeros((2, 7, 7, 30))
    print('x.unsqueeze(-1) shape: ', x.unsqueeze(-1).shape)
    print('x.unsqueeze(0) shape: ', x.unsqueeze(0).shape)
    print('x.unsqueeze(1) shape: ', x.unsqueeze(1).shape)


def torch_cat():
    x1 = torch.zeros((2, 3, 3, 5))
    x2 = torch.zeros((4, 3, 3, 5))
    output = torch.cat([x1, x2], dim=0)
    print('torch.cat([x1, x2], dim=0):', output.shape)


def tensor_argmax():
    x = torch.rand((3, 1, 4, 3))
    output = x.argmax(0)
    print('x.argmax(0) shape: ', output.shape)
    print('x.argmax(0): ', output)


def tensor_repeat():
    output = torch.arange(7).repeat(2, 7, 1).unsqueeze(-1)
    print('output shape: ', output.shape)
    print('output: ', output)


def tensor_permute():
    x = torch.arange(7).repeat(2, 7, 1).unsqueeze(-1)
    print('x shape: ', x.shape)
    print('x: ', x)
    output = x.permute(0, 2, 1, 3)
    print('output shape: ', output.shape)
    print('output: ', output)


def torch_flatten():
    x = torch.zeros((2, 7, 7, 4))
    output = torch.flatten(x, end_dim=-2)
    print('output shape:', output.shape)


if __name__ == '__main__':
    torch_flatten()
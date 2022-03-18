import torch
import torch.nn as nn

architecture_config = [
    # layer = (out_channels, kernel size, padding, stride)
    # input (3,448,448)
    (64, 7, 3, 2),  # (64,224,224) (448-7+3*2)/2+1=224 discard the last element
    'MP',  # max pooling 2x2 stride=2 -> (64,112,112) 224/2=112
    (192, 3, 1, 1),  # (192,112,112) (112-3+1*2)/1+1=112
    'MP',  # max pooling 2x2 stride=2 -> (192,56,56)
    (128, 1, 0, 1),
    (256, 3, 1, 1),
    (256, 1, 0, 1),
    (512, 3, 1, 1),  # (512,56,56)
    'MP',  # max pooling 2x2 stride=2 -> (512,28,28)
    # [layer1, layer2, repeat times]
    [(256, 1, 0, 1), (512, 3, 1, 1), 4],
    (512, 1, 0, 1),
    (1024, 3, 1, 1),  # (1024,28,28)
    'MP',  # max pooling 2x2 stride=2 -> (1024,14,14)
    [(512, 1, 0, 1), (1024, 3, 1, 1), 2],
    (1024, 3, 1, 1),
    (1024, 3, 1, 2),  # (1024,7,7) (14-3+1*2)/2+1=7 discard the last element
    (1024, 3, 1, 1),
    (1024, 3, 1, 1)
]


class ConvBlock(nn.Module):
    """
        CBL Convolution BatchNormalization LeakyRelu
        Actually, YOLOv1 don't have batchNormalization.
    """

    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        # batch normalization
        self.bn = nn.BatchNorm2d(out_channels)
        # activation function
        self.af = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.af(x)
        return x


class YOLO(nn.Module):
    """
        YOLOv1 backbone (You Only Look Once)
        more detail about network architecture in /img/YOLOv1_backbone1.png
    """

    def __init__(self, in_channels=3, **kwargs):
        super(YOLO, self).__init__()
        self.arch_conf = architecture_config
        self.in_channels = in_channels
        self.backbone = self._create_conv_layers(architecture_config)
        self.fc = self._create_fc(**kwargs)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

    def _create_conv_layers(self, arch):
        blk = []
        in_channels = self.in_channels

        for layer in arch:
            if type(layer) == tuple:
                blk.append(
                    ConvBlock(
                        in_channels, layer[0], kernel_size=layer[1],
                        padding=layer[2], stride=layer[3]
                    )
                )

                in_channels = layer[0]

            elif type(layer) == 'MP':
                blk.append(
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )

            elif type(layer) == list:
                num_repeat = layer[2]
                conv1 = layer[0]
                conv2 = layer[1]

                for _ in range(num_repeat):
                    blk.append(
                        ConvBlock(
                            in_channels, conv1[0], kernel_size=conv1[1],
                            padding=conv1[2], stride=conv1[3]
                        )
                    )

                    blk.append(
                        ConvBlock(
                            in_channels, conv2[0], kernel_size=conv2[1],
                            padding=conv2[2], stride=conv2[3]
                        )
                    )

                in_channels = conv2[0]

        return nn.Sequential(*blk)

    def _create_fc(self, num_grid_cell, num_bbox, num_class):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 7 * 7, 4096),
            nn.Dropout(0.6),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, num_grid_cell * num_grid_cell * (5 * num_bbox + num_class))
        )


def test():
    model = YOLO(num_grid_cell=7, num_bbox=2, num_class=20)
    x = torch.randn((1, 3, 448, 448))
    print(model(x).shape)



if __name__ == '__main__':
    test()
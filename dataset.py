import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import utils
from torchvision.transforms import ToTensor


class BananasDataset(torch.utils.data.Dataset):
    """
        Banana detection dataset
        It is from https://www.bilibili.com/video/BV1Lh411Y7LX?p=3, which is made by Li Mu and his mates.
        you can download it from http://d2l-data.s3-accelerate.amazonaws.com/banana-detection.zip
    """
    def __init__(self, dataset_path, S=7, B=2, C=1, transform=None):
        self.S = S
        self.B = B
        self.C = C
        self.transform = transform
        self.img_dir = os.path.join(dataset_path, 'images')
        self.labels = pd.read_csv(os.path.join(dataset_path, 'label.csv'))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # an image have a banana
        class_label = int(self.labels.iloc[index]['label'])
        # [xmin, ymin, xmax, ymax]
        box = self.labels.iloc[index, 2:].values.tolist()

        img_path = os.path.join(self.img_dir, self.labels.iloc[index]['img_name'])
        image = Image.open(img_path)

        # convert to [x, y, width, height]
        box = utils.box_corner_to_center(box)
        # The size of the image is 256x256
        box = torch.tensor(box) / 256.0

        if self.transform:
            image = self.transform(image)

        label = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        x, y, width, height = box.tolist()
        i, j = int(self.S * x), int(self.S * y)
        x_cell, y_cell = self.S * x - i, self.S * y - j
        width_cell, height_cell = (
            width * self.S,
            height * self.S
        )
        #
        if label[j, i, 1] == 0:
            label[j, i, 1] = 1  # obj confidence is 1
            box_coordinates = torch.tensor(
                [x_cell, y_cell, width_cell, height_cell]
            )
            # only assign coordinate to the first box
            # the second box is zeros
            label[j, i, 2:6] = box_coordinates
            label[j, i, class_label] = 1
        return image, label


def test_bananas_dataset_1():
    trainDataset = BananasDataset('data/banana-detection/bananas_val/')
    print('train data num: ', len(trainDataset))
    # select the first group of train data
    train_data = trainDataset[0]
    print(train_data[1].shape)
    print(train_data[0])
    train_data[0].show()


def test_bananas_dataset_2():
    train_dataset = BananasDataset('data/banana-detection/bananas_val/', transform=ToTensor())
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=16,
        num_workers=4,
        pin_memory=True,
        shuffle=True,
        drop_last=True
    )
    for (x, y) in train_loader:
        print('image shape: ', x.shape)
        print('label shape: ', y.shape)
        break


if __name__ == '__main__':
    test_bananas_dataset_1()

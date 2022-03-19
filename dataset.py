import torch
import os
import pandas as pd
from PIL import Image
from utils import box_corner_to_center


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
        self.img_dir = os.path.join(dataset_path, 'banana-detection/bananas_train/images')
        self.labels = pd.read_csv(os.path.join(dataset_path, 'banana-detection/bananas_train/label.csv'))

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
        box = box_corner_to_center(box)
        # The size of the image is 256x256
        box = torch.tensor(box) / 256.0

        if self.transform:
            image, box = self.transform(image, box)

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
            label[j, i, 2:6] = box_coordinates
            label[j, i, class_label] = 1
        return image, label


def test():
    trainDataset = BananasDataset('data')
    print('train data size: ', len(trainDataset))
    # select the first group of train data
    train_data = trainDataset[0]
    print(train_data[1].shape)
    train_data[0].show()


if __name__ == '__main__':
    test()


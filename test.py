from PIL import Image
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from model import YOLO
from dataset import BananasDataset
from utils import (
    nms,
    plot_image,
    cellboxes_to_boxes
)


def main():
    model = YOLO(S=7, B=2, C=1)
    # load model from disk
    model.load_state_dict(torch.load('./doc/yolov1-weight-2.pt'))
    model.eval()

    transform = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor()])
    test_dataset = BananasDataset('data/banana-detection/bananas_val/', transform=transform)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=4,
        num_workers=1,
        pin_memory=True,
        shuffle=True,
        drop_last=True
    )

    for batch_idx, (x, y) in enumerate(test_loader):
        if batch_idx == 1:
            break

        batch_size = x.shape[0]
        predictions = model(x)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            if idx == 1:
                break

            nms_boxes = nms(
                bboxes[idx],
                iou_threshold=0.5,
                threshold=0.2,
                box_mode='midpoint'
            )

            plot_image(x[idx].permute(1, 2, 0).to('cpu'), nms_boxes, 'midpoint')


def detect(img_path):
    img = Image.open(img_path)

    model = YOLO(S=7, B=2, C=1)
    # load model from disk
    model.load_state_dict(torch.load('./doc/yolov1-weight-2.pt'))
    # evaluation mode
    model.eval()

    transform = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor()])
    x = transform(img)

    x = x.reshape((1, 3, 448, 448))
    predictions = model(x)
    bboxes = cellboxes_to_boxes(predictions)

    nms_boxes = nms(
        bboxes[0],
        # 0.5
        iou_threshold=0.8,
        threshold=0.3,
        box_mode='midpoint'
    )

    plot_image(x[0].permute(1, 2, 0).to('cpu'), nms_boxes, 'midpoint')


if __name__ == '__main__':
    # test4 test1 iou_threshold=0.8
    detect('./img/test/test2.png')
import utils
import torch


def test_iou_1():
    boxes_pred = torch.tensor([0.2, 0.2, 0.4, 0.4])
    boxes_label = torch.tensor([0.3, 0.4, 0.6, 0.6])
    mode = 'corner'
    iou_values = utils.iou(boxes_pred, boxes_label, mode)
    print(iou_values)


def test_iou_2():
    boxes_pred = torch.tensor([0.3, 0.3, 0.2, 0.2])
    boxes_label = torch.tensor([0.5, 0.5, 0.4, 0.4])
    mode = 'midpoint'
    iou_values = utils.iou(boxes_pred, boxes_label, mode)
    print(iou_values)


if __name__ == '__main__':
    test_iou_1()
import logging
import torch
from torchsummary import summary


class Logger:
    def __init__(self, name=__name__):
        self.__name = name
        self.logger = logging.getLogger(self.__name)
        self.logger.setLevel(logging.DEBUG)

        # create a handler, print log info to console
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # define format
        formatter = logging.Formatter('%(asctime)s %(filename)s-[line:%(lineno)d]'
                                      '-%(levelname)s-[%(name)s]: %(message)s',
                                      datefmt='%a, %d %b %Y %H:%M:%S')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    @property
    def get_log(self):
        return self.logger


# TODO: 删除这个函数 放在YOLO类中
def model_info(model):
    summary(model, (3, 448, 448), batch_size=1, device="cpu")


def iou(boxes_pred, boxes_label, mode='midpoint'):
    """
        Calculates intersection over union
        Parameters:
            boxes_pred (tensor): bounding boxes (batch_size, 4)
            boxes_label (tensor): ground truth boxes (batch_size, 4)
            mode (str): midpoint(x,y,w,h) / corner (x1,y1,x2,y2)
        Returns:
            tensor: intersection over union (batch_size, 1)
    """

    # convert
    if mode == 'midpoint':
        # more detail about ... in difficulties.py(three_dots)
        boxes_pred_x1 = boxes_pred[..., 0:1] - boxes_pred[..., 2:3] / 2
        # left top is the coordinate origin!
        boxes_pred_y1 = boxes_pred[..., 1:2] - boxes_pred[..., 3:4] / 2
        boxes_pred_x2 = boxes_pred[..., 0:1] + boxes_pred[..., 2:3] / 2
        boxes_pred_y2 = boxes_pred[..., 1:2] + boxes_pred[..., 3:4] / 2
        boxes_label_x1 = boxes_label[..., 0:1] - boxes_label[..., 2:3] / 2
        boxes_label_y1 = boxes_label[..., 1:2] - boxes_label[..., 3:4] / 2
        boxes_label_x2 = boxes_label[..., 0:1] + boxes_label[..., 2:3] / 2
        boxes_label_y2 = boxes_label[..., 1:2] + boxes_label[..., 3:4] / 2

    elif mode == 'corner':
        boxes_pred_x1 = boxes_pred[..., 0:1]
        boxes_pred_y1 = boxes_pred[..., 1:2]
        boxes_pred_x2 = boxes_pred[..., 2:3]
        boxes_pred_y2 = boxes_pred[..., 3:4]
        boxes_label_x1 = boxes_label[..., 0:1]
        boxes_label_y1 = boxes_label[..., 1:2]
        boxes_label_x2 = boxes_label[..., 2:3]
        boxes_label_y2 = boxes_label[..., 3:4]

    else:
        return

    # Calculates intersection
    # more detail in img/torch_max.png and difficulties.py(torch_max)
    x_top_left = torch.max(boxes_pred_x1, boxes_label_x1)
    y_top_left = torch.max(boxes_pred_y1, boxes_label_y1)
    x_bottom_right = torch.max(boxes_pred_x2, boxes_label_x2)
    y_bottom_right = torch.max(boxes_pred_y2, boxes_label_y2)

    # clamp(0) will modify the element to 0 if it less than 0
    intersection = (x_bottom_right - x_top_left).clamp(0) * (y_bottom_right - y_top_left).clamp(0)

    boxes_pred_area = abs((boxes_pred_x2 - boxes_pred_x1) * (boxes_pred_y2 - boxes_pred_y1))
    boxes_label_area = abs((boxes_label_x2 - boxes_label_x1) * (boxes_label_y2 - boxes_label_y1))

    return intersection / (boxes_pred_area + boxes_label_area - intersection + 1e-6)


if __name__ == '__main__':
    log = Logger('model').get_log
    log.info('this is a test')
    logging.info(123)

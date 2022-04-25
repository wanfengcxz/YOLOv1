import logging
import torch
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt


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


def box_corner_to_center(boxes):
    """从（左上，右下）转换到（中间，宽度，高度） (list)"""
    x1, y1, x2, y2 = boxes[0], boxes[1], boxes[2], boxes[3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = [cx, cy, w, h]
    return boxes


def iou(boxes_pred, boxes_label, box_mode='midpoint'):
    """
    Calculate intersection over union.
    Args:
        boxes_pred (tensor): bounding boxes (..., 4)
        boxes_label (tensor): ground truth boxes (...., 4)
        box_mode (str): midpoint(x,y,w,h) / corner (x1,y1,x2,y2)

    Returns:
        iou: (..., 4)
    """
    # convert
    if box_mode == 'midpoint':
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

    elif box_mode == 'corner':
        # print('corner')
        boxes_pred_x1 = boxes_pred[..., 0:1]
        boxes_pred_y1 = boxes_pred[..., 1:2]
        boxes_pred_x2 = boxes_pred[..., 2:3]
        boxes_pred_y2 = boxes_pred[..., 3:4]
        boxes_label_x1 = boxes_label[..., 0:1]
        boxes_label_y1 = boxes_label[..., 1:2]
        boxes_label_x2 = boxes_label[..., 2:3]
        boxes_label_y2 = boxes_label[..., 3:4]

    # Calculates intersection
    # more detail in img/torch_max.png and difficulties.py(torch_max)
    x_top_left = torch.max(boxes_pred_x1, boxes_label_x1)
    y_top_left = torch.max(boxes_pred_y1, boxes_label_y1)
    x_bottom_right = torch.min(boxes_pred_x2, boxes_label_x2)
    y_bottom_right = torch.min(boxes_pred_y2, boxes_label_y2)
    # print(x_top_left,y_top_left)
    # print(x_bottom_right,y_bottom_right)

    # clamp(0) will modify the element to 0 if it less than 0
    intersection = (x_bottom_right - x_top_left).clamp(0) * (y_bottom_right - y_top_left).clamp(0)

    boxes_pred_area = abs((boxes_pred_x2 - boxes_pred_x1) * (boxes_pred_y2 - boxes_pred_y1))
    boxes_label_area = abs((boxes_label_x2 - boxes_label_x1) * (boxes_label_y2 - boxes_label_y1))

    return intersection / (boxes_pred_area + boxes_label_area - intersection + 1e-6)


def plot_image(image, boxes, box_mode, box_color='w'):
    """
    Plot the bounding box in the image.
    Args:
        box_mode: midpoint
        image: image
        boxes(list): the coordinate is [class_pred, obj_confidence, x, y, width, height] and its values is between 0 and 1.
        box_color: the color of the rectangle
    """

    img = np.array(image)
    height, width, _ = img.shape
    fig = plt.imshow(img)

    for box in boxes:
        score = box[1]
        box = box[2:]
        if box_mode == 'corner':
            box = box_corner_to_center(box)
        assert len(box) == 4, 'Got more values than in x, y, w, h, in a box!'
        top_left_x = box[0] - box[2] / 2
        top_left_y = box[1] - box[3] / 2
        rect = plt.Rectangle(
            (top_left_x * width, top_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor=box_color,
            facecolor='none'
        )
        fig.axes.add_patch(rect)
        fig.axes.text(top_left_x * width, top_left_y * height, '%.2f' % score,
                      bbox={'facecolor': 'white', 'alpha': 0.9})

    plt.show()


def get_bboxes(
        loader,
        model,
        iou_threshold,
        threshold,
        box_mode='midpoint',
        device='cuda'
):
    all_pred_boxes = []
    all_true_boxes = []

    model.eval()
    train_img_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        predictions.to('cpu')

        # converts its ratios and its type(tensor to list)
        # shape is (batch_size, S*S, 6)
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = nms(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_mode=box_mode
            )

            # if batch_idx == 0 and idx == 0:
            #    plot_image(x[idx].permute(1,2,0).to("cpu"), nms_boxes)
            #    print(nms_boxes)

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_img_idx] + nms_box)
            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_img_idx] + box)

            train_img_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes


def nms(bboxes, iou_threshold, threshold, box_mode):
    """
    Does Non Max Suppression given bboxes
    Args:
        bboxes(list): its shape is (S*S, 6) [class_pred, obj_confidence, x1, y1, x2, y2]
        iou_threshold(float): threshold where predicted bboxes is correct
        threshold(float): threshold to remove predicted bboxes (independent of IoU)
        box_mode(str): "midpoint" or "corner"
    Returns:
        bboxes_after_nms(list): (..., 6)
    """
    assert type(bboxes) == list

    bboxes = [bbox for bbox in bboxes if bbox[1] > threshold]
    # Sort from little to big
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            # not same class or iou < threshold
            if box[0] != chosen_box[0]
               or iou(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_mode=box_mode
            ) < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def convert_cellboxes(predictions, C):
    """
    Converts bounding boxes output from YOLO with relative to grid cell ratios
    into the entire image ratios in a vectorization method.
    It is difficult to read code.
    Args:
        predictions(tensor): its shape is (batch_size, S, S, B * 5 + C)
        C: num of class

    Returns:
        converted_pred(tensor): its shape is (batch_size, S, S, 6)
    """
    S = 7
    B = 2
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, S, S, B * 5 + C)

    # bbox shape: (batch_size, S, S, 4)
    # [x, y, width, height]
    bbox1 = predictions[..., C + 1:C + 5]
    bbox2 = predictions[..., C + 6:C + 10]

    """
        get object confidence
        scores shape: (B, batch_size, S, S)
        others: pay more attention to predictions[..., 1], single '1' is get the element,
        not list, actually, predictions[..., 1:2] is get the list, more detail about 
        this in difficulties.py(three_dots)
    """
    scores = torch.cat(
        (predictions[..., C].unsqueeze(0), predictions[..., C + 5].unsqueeze(0)), dim=0)

    """"
        对第一个维度（即两组bbox,bbox1和bbox2）求出最大置信度的bbox所在的下标，要么为0要么为1
        you can see more detail in img/best_box.png
        best_box shape: (batch_size, S, S, 1)
        others: more detail about unsqueeze in difficulties.py(tensor_unsqueeze),
        more detail about argmax in difficulties.py(tensor_argmax)
    """
    best_box = scores.argmax(0).unsqueeze(-1)

    # filter out bbox with low confidence
    # best_boxes shape: (batch_size, S, S, 4)
    best_boxes = bbox1 * (1 - best_box) + best_box * bbox2

    # more detail about repeat in difficulties.py(tensor_repeat)
    # cell_indices shape: (batch_size, S, S, 1)
    cell_indices = torch.arange(7).repeat(batch_size, S, 1).unsqueeze(-1)

    # convert x,y into entire image ratios rather than relative to grid cell ratios
    x = 1 / S * (best_boxes[..., 0:1] + cell_indices)
    # more detail about permute in img/tensor_permute.png
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    width_height = 1 / S * best_boxes[..., 2:4]

    # converted_bbox shape: (batch_size, S, S, 4)
    converted_bbox = torch.cat((x, y, width_height), dim=-1)

    # predicted_class shape: (batch_size, S, S, 1)
    predicted_class = predictions[..., 0:C].argmax(-1).unsqueeze(-1)
    # best_confidence shape: (batch_size, S, S, 1)
    best_confidence = torch.max(predictions[..., C], predictions[..., C + 5]).unsqueeze(-1)
    # converted_pred shape: (batch_size, S, S, C+5)
    converted_pred = torch.cat(
        (predicted_class, best_confidence, converted_bbox), dim=-1
    )
    return converted_pred


def cellboxes_to_boxes(out, S=7):
    """
    Converts tensor to list.
    Args:
        out(tensor): its shape is (batch_size, S, S, B * 5 + C)
        S: const value(7)

    Returns:
        all_boxes(list): its shape is (batch_size,S*S,C+5)
    """
    # converted_pred shape: (batch_size, S*S, 6)
    converted_pred = convert_cellboxes(out, 1).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    batch_size = out.shape[0]
    for idx in range(batch_size):
        bboxes = []
        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    # all_boxes shape: (batch_size,S*S,6)
    return all_bboxes


# def mean_average_precision(
#         pred_boxes, true_boxes, iou_threshold=0.5, box_mode='midpoint', num_classes=1
# ):
#     """
#     Calculates mean average precision
#     Args:
#         pred_boxes (list):
#         true_boxes:
#         iou_threshold:
#         box_mode:
#         num_classes:
#
#     Returns:
#         mAP value
#     """
#     # list for storing all AP for respective classes
#     average_precisions = []
#
#     # used for numerical stability later on
#     epsilon = 1e-6
#
#     for c in range(num_classes):
#         detections = []
#         ground_truths = []
#
#         for detection in pred_boxes:
#             if detection[1] == c:
#                 detections.append(detection)
#
#         for true_box in true_boxes:
#             if true_box[1] == c:
#                 ground_truths.append(true_box)
#
#
#
#

if __name__ == '__main__':
    log = Logger('model').get_log
    log.info('this is a test')
    logging.info(123)

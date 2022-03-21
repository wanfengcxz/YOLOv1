import torch
import torch.nn as nn
from utils import iou


class YOLOLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YOLOLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = 5
        self.lambda_noobj = 0.5

    def forward(self, predictions, labels):
        """
        Calculate loss
        Args:
            predictions: dim: (batch_size, S*S*(B*5+C))
            labels: dim: (batch_size, S, S, (B*5+C))
        """
        # predictions shape: (batch_size, S, S, (B*5+C))
        predictions = predictions.reshape(-1, self.S, self.S, self.B * 5 + self.C)

        # every grid cell generate two bounding boxes
        # iou_bx shape: (batch_size, S, S, 1)
        iou_b1 = iou(predictions[..., self.C + 1:self.C + 5], labels[..., self.C + 1:self.C + 5])
        iou_b2 = iou(predictions[..., self.C + 6 + self.C + 10], labels[..., self.C + 1:self.C + 5])

        # ious shape: (2, batch_size, S, S, 1)
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        # To two bbox of every grid cell, find the best bbox
        # iou_max shape: (batch_size, S, S, 1)
        # best_box_index shape: (batch_size, S, S, 1)
        iou_max, best_box_index = torch.max(ious, dim=0)
        # exists_box shape: (batch_size, S, S, 1)
        exists_box = labels[..., self.C].unsqueeze(3)  # obj

        # ------------------------------
        #   coordinate regression loss
        # ------------------------------

        # box_predictions shape: (batch_size, S, S, 4)
        box_predictions = exists_box * (
                best_box_index * predictions[..., self.C + 6:self.C + 10]
                + (1 - best_box_index) * predictions[..., self.C + 1:self.C + 5]
        )

        # box_labels shape: (batch_size, S, S, 4)
        box_labels = exists_box * labels[..., self.C + 1:self.C + 5]

        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6))

        box_labels = torch.sqrt(box_labels[..., 2:4])

        # more detail about torch.flatten in difficulties.py(torch_flatten)
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_labels, end_dim=-2)
        )

        # -------------------------------------
        #   object confidence regression loss
        # -------------------------------------

        # pred_box shape: (batch_size, S, S, 1)
        pred_box = (
                best_box_index * predictions[..., self.C + 5:self.C + 6] +
                (1 - best_box_index) * predictions[..., self.C:self.C + 1]
        )

        #
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * labels[..., self.C:self.C + 1] * iou_max)
        )

        # (batch_size, S*S)
        no_obj_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., self.C:self.C + 1], start_dim=1),
            torch.flatten((1 - exists_box) * labels[..., self.C:self.C + 1], start_dim=1)
        )

        no_obj_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., self.C + 5:self.C + 6], start_dim=1),
            torch.flatten((1 - exists_box) * labels[..., self.C:self.C + 1], start_dim=1)
        )

        # -------------------------------------
        #   class probability regression loss
        # -------------------------------------

        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :self.C], end_dim=-2),
            torch.flatten(exists_box * labels[..., :self.C], end_dim=-2)
        )

        loss = (
                self.lambda_coord * box_loss
                + object_loss
                + self.lambda_noobj * no_obj_loss
                + class_loss
        )

        return loss

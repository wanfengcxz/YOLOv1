import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import gc
import utils
from model import YOLO
from loss import YOLOLoss
from dataset import BananasDataset

LEARNING_RATE = 2e-5
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 4
WEIGHT_DECAY = 0
EPOCHS = 15
PIN_MEMORY = True
LOAD_MODEL = False
NUM_WORKERS = 2


def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        # print(x.shape, y.shape)
        out = model(x)
        # print(out.shape)
        loss = loss_fn(out, y)
        # tensor.item() return a number
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss) / len(mean_loss)}")


def main():
    transform = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor()])
    model = YOLO(S=7, B=2, C=1).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YOLOLoss(S=7, B=2, C=1)

    train_dataset = BananasDataset('data', transform=transform)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True
    )

    print('start train')
    for epoch in range(EPOCHS):
        # pred_boxes, target_boxes = utils.get_bboxes(
        #     train_loader, model, iou_threshold=0.5, threshold=0.4
        # )

        print(f'epoch:{epoch + 1}')
        train_fn(train_loader, model, optimizer, loss_fn)
        gc.collect()
        torch.cuda.empty_cache()
    torch.save(model.state_dict(), './doc/yolov1-weight-2.pt')


if __name__ == '__main__':
    main()

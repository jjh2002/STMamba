import datetime
import torch, gc
import torch.optim as optim
from torch.optim import lr_scheduler

from utils import *
import random
from tqdm import tqdm
import torch.nn as nn
from model import Mamba, MambaConfig
from torch.utils.data import DataLoader, TensorDataset
from early_stopping import EarlyStopping
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--use_cuda', default=True,
                    help='CUDA training.')
parser.add_argument('--seed', type=int, default=2024, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.002,
                    help='Learning rate.')
parser.add_argument('--wd', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8,
                    help='Dimension of representations')
parser.add_argument('--layer', type=int, default=2,
                    help='Num of layers')
parser.add_argument('--predL', type=int, default=48,
                    help='length of predictions')
parser.add_argument('--num_nodes', type=int, default=325,
                    help='num of nodes')
parser.add_argument('--HisL', type=int, default=144,
                    help='length of history')

args = parser.parse_args()

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)

device = torch.device("cuda:0" if args.use_cuda else "cpu")

model_args = MambaConfig(d_model=args.hidden, n_layers=args.layer, HisL=args.HisL, num_nodes=args.num_nodes, predL=args.predL)
model = Mamba(model_args).to(device)

# Initialize weights
def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.uniform_(m.weight, -0.1, 0.1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


model.apply(init_weights)
# model.load_state_dict(torch.load('pemsbay.pth'))
# Data preparation
datatr = []
datate = []
# BJ PM2.5
# datatr = np.load('/kaggle/input/stdata/x_1hour_10day.npy')
# datate = np.load('/kaggle/input/stdata/y_1hour_3day.npy')
# num_train = 642
# num_valid = 214
# num_test = 214
# PEMS-BAY
# datatr = np.load('/kaggle/input/st-mamba-data/PEMS-BAY/x_30min.npy')
# datate = np.load('/kaggle/input/st-mamba-data/PEMS-BAY/y_30min.npy')
# num_train = 850
# num_valid = 283
# num_test = 283
num_train = 6
num_valid = 2
num_test = 2
datatr = np.random.rand(10, 144, 325, 4)
datate = np.random.rand(10, 48, 325, 4)
# METR-LA
# datatr = np.load('/kaggle/input/st-mamba-data/METR-LA/x_30min.npy')
# datate = np.load('/kaggle/input/st-mamba-data/METR-LA/y_30min.npy')
# num_train = 553
# num_valid = 184
# num_test = 184


trainX = torch.from_numpy(datatr[:num_train, :, :, :]).float().to(device)
trainY = torch.from_numpy(datate[:num_train, :, :, :1]).float().to(device)
validX = torch.from_numpy(datatr[num_train:(num_train + num_valid), :, :, :]).float().to(device)
validY = torch.from_numpy(datate[num_train:(num_train + num_valid), :, :, :1]).float().to(device)
testX = torch.from_numpy(datatr[(num_train + num_valid):, :, :, :]).float().to(device)
testY = torch.from_numpy(datate[(num_train + num_valid):, :, :, :1]).float().to(device)

# Create DataLoader
batch_size = 1
train_dataset = TensorDataset(trainX, trainY)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
valid_dataset = TensorDataset(validX, validY)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_dataset = TensorDataset(testX, testY)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Optimizer and learning rate scheduler
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.wd)
scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.8)

# Early stopping
early_stopping = EarlyStopping(patience=20, verbose=True)
def train_one_epoch(epoch, model, optimizer, train_loader):
    model.train()
    train_losses = []
    train_dlosses = []
    for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch} - Training"):
        # Assuming get_batch_feed_dict handles batching
        # preds, labels, sloss, eF, dF = model(x)
        preds, labels, sloss = model(x_batch, y_batch)
        # Train discriminator
        # optimizer_D.zero_grad()
        #
        # outputs_real = discriminator(eF.permute(2, 0, 1).detach())
        # d_loss_real = criterion_GAN(outputs_real, real_labels)
        #
        # outputs_fake = discriminator(dF.permute(2, 0, 1).detach())
        # d_loss_fake = criterion_GAN(outputs_fake, fake_labels)
        # d_loss = d_loss_real+d_loss_fake
        # d_loss.backward()
        # optimizer_D.step()
        # print(f"Epoch {epoch}, D-Real Loss: {d_loss_real.item()}, D-Fake Loss: {d_loss_fake.item()}")
        # Train generator
        optimizer.zero_grad()
        # outputs = discriminator(dF.permute(2, 0, 1))
        # g_loss = criterion_GAN(outputs, real_labels)

        # loss = masked_mae(preds.float(), labels.float()) + sloss + g_loss
        loss = masked_mae(preds.float(), labels.float()) + sloss
        loss.backward(retain_graph=True)  # 保留计算图以便下次反向传播
        optimizer.step()

        train_losses.append(loss.item())
        # train_dlosses.append(d_loss.item())
    return np.average(train_losses)


def validate(model, valid_loader):
    model.eval()
    valid_losses = []
    with torch.no_grad():
        for x_batch, y_batch in tqdm(valid_loader, desc="Validation"):
            # Assuming get_batch_feed_dict handles batching
            # preds, labels, sloss, eF, dF = model(x)
            preds, labels, sloss = model(x_batch, y_batch)
            # outputs = discriminator(dF.permute(2, 0, 1))
            # g_loss = criterion_GAN(outputs, real_labels)
            # loss = masked_mae(preds.float(), labels.float()) + sloss + g_loss
            loss = masked_mae(preds.float(), labels.float())
            valid_losses.append(loss.item())
    return np.average(valid_losses)


def test(model, test_loader):
    model.eval()
    test_losses, test_losses1 = [], []
    with torch.no_grad():
        for x_batch, y_batch in tqdm(test_loader, desc="Testing"):
         # Assuming get_batch_feed_dict handles batching
            # preds, labels, _, eF, dF = model(x)
            preds, labels, _ = model(x_batch,y_batch)
            test_losses.append(masked_mae(preds.float(), labels.float()).item())
            test_losses1.append(masked_rmse(preds.float(), labels.float()).item())
    return np.average(test_losses), np.average(test_losses1)


print('开始时间-------', datetime.datetime.now())
total_epoch = args.epochs
train_loss, valid_loss, test_mae, test_rmse = [], [], [], []
avg_train_losses, avg_valid_losses = [], []
test_mae, test_rmse = 0, 0
for epoch in range(total_epoch):
    print(f'----------epoch {epoch}-----------{datetime.datetime.now()}')

    train_loss = train_one_epoch(epoch, model, optimizer, train_loader)
#     valid_loss1 = validate(model, train_loader)
    valid_loss = validate(model, valid_loader)

    avg_train_losses.append(train_loss)
    avg_valid_losses.append(valid_loss)

    early_stopping(valid_loss, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break

    scheduler.step()

    test_mae, test_rmse = test(model, test_loader)
    print(
        f'Train Loss: {train_loss:.6f}, Valid Loss: {valid_loss:.6f}, Test MAE: {test_mae:.6f}, Test RMSE: {test_rmse:.6f}')
    test_mae, test_rmse = test(model, test_loader)
    gc.collect()
    # torch.save(model.state_dict(), 'woslicebj.pth')
    torch.cuda.empty_cache()
print('===============METRIC===============')
print(f'MAE = {test_mae:.6f}')
print(f'RMSE = {test_rmse:.6f}')
print('结束时间-------', datetime.datetime.now())

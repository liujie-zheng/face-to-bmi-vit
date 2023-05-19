import torch
import torch.nn as nn
from loader import get_dataloaders
from models import get_model

import numpy as np
import argparse


# train one epoch
def train(train_loader, model, loss_fn, optimizer):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Train
    model.train()
    for batch, (X, y) in enumerate(train_loader):
        X = X.to(device)
        y = y.to(device)

        # Compute prediction error
        pred = model(X)
        y = y.unsqueeze(1).float()
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Show progress
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"train loss: {loss:>7f} [{current:>5d}/{len(train_loader.dataset):>5d}]")


# validate and return mae loss
def validate(val_loader, model):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Validation
    model.eval()
    val_loss_mse = 0
    val_loss_mae = 0
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(val_loader):
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            y = y.unsqueeze(1)

            loss_mse = nn.MSELoss()(pred, y)
            val_loss_mse += loss_mse.item()
            loss_mae = nn.L1Loss()(pred, y)
            val_loss_mae += loss_mae.item()

    val_loss_mse /= len(val_loader)
    val_loss_mae /= len(val_loader)

    print(f"val mse loss: {val_loss_mse:>7f}, val mae loss: {val_loss_mae}")
    return val_loss_mae



# test and return mse and mae loss
def test(test_loader, model):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Test
    model.eval()
    test_loss_mse = 0
    test_loss_mae = 0
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(test_loader):
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            y = y.unsqueeze(1)

            loss_mse = nn.MSELoss()(pred, y)
            test_loss_mse += loss_mse.item()
            loss_mae = nn.L1Loss()(pred, y)
            test_loss_mae += loss_mae.item()

    test_loss_mse /= len(test_loader)
    test_loss_mae /= len(test_loader)

    print(f"test mse loss: {test_loss_mse:>7f}, test mae loss: {test_loss_mae}")
    return test_loss_mse, test_loss_mae



# helper class for early stopping
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), '../weights/checkpoint.pt')  # save checkpoint
        self.val_loss_min = val_loss



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument('--augmented', type=bool, default=False, help='set to True to use augmented dataset')
    args = parser.parse_args()

    train_loader, val_loader, test_loader = get_dataloaders(16, augmented=args.augmented, vit_transformed=True, show_sample=True)
    model = get_model().float().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 50
    early_stopping = EarlyStopping(patience=5, verbose=True)

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer)
        val_loss = validate(test_loader, model)
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # model.load_state_dict(torch.load('../weights/no_aug_epoch_10.pt'))
    test(test_loader, model)

    print("Done!")



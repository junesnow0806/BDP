import torch
from torch import nn

from resnet20 import resnet20
from read_data import read_cifar10

import os
import time

def train(model, optimizer, train_loader, n_epoch=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1, last_epoch=-1)

    training_loss_list = []
    training_acc_list = []
    n_step = len(train_loader)
    for epoch in range(n_epoch):
        model.train()
        training_loss = 0.0
        since = time.time()
        
        print(f"Epoch {epoch+1}/{n_epoch}")
        print("-"*30)

        total = 0
        correct = 0
        for i, data in enumerate(train_loader):
            x, labels = data
            x, labels = x.to(device), labels.to(device)

            # forward
            outputs = model(x)
            loss = criterion(outputs, labels)
            training_loss += loss.item()

            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 99:
                print(f"Epoch: {epoch+1}/{n_epoch}, Step: {i+1}/{n_step}, Loss: {loss.item():.4f}")
        training_loss /= n_step
        training_loss_list.append(training_loss)
        training_acc = 100 * correct / total
        training_acc_list.append(training_acc)
        print(f"Epoch Average Loss: {training_loss}, Training ACC: {training_acc}%")
        scheduler.step()

        time_used = time.time() - since
        print(f"Epoch used time: {time_used // 60}min, {time_used % 60}sec")
    
    return model, training_loss_list, training_acc_list


def test(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            x, labels = data
            x, labels = x.to(device), labels.to(device)
            outputs = model(x)
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    print(f"Test ACC: {100 * correct / total}%")


def save(save_dir, model, training_loss, training_acc):
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))
    with open(os.path.join(save_dir, "training_loss.txt"), "w") as f:
        f.write(str(training_loss))
    with open(os.path.join(save_dir, "training_acc.txt"), "w") as f:
        f.write(str(training_acc))


def load(load_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet20()
    model.load_state_dict(torch.load(load_path))
    model = model.to(device)
    return model


if __name__ == "__main__":
    train_loader, test_loader = read_cifar10(128, "data/")

    model = resnet20()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # trained_model, training_loss_list, training_acc_list = train(model, optimizer, train_loader)
    trained_model = load("results/resnet20.pth")
    test(trained_model, test_loader)

    # save(trained_model, "save/resnet20/")
    # with open("results/resnet20_training_loss.txt", "w") as f:
    #     f.write(str(training_loss_list))
    # with open("results/resnet20_training_acc.txt", "w") as f:
    #     f.write(str(training_acc_list))
     

import torch

from resnet20 import resnet20
from read_data import read_cifar10

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

import warnings
warnings.simplefilter("ignore")

def load_private_model(load_path, train_loader, n_epoch, epsilon, delta, c):
    model = resnet20()
    model = ModuleValidator.fix(model)
    ModuleValidator.validate(model, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=n_epoch,
        target_epsilon=epsilon,
        target_delta=delta,
        max_grad_norm=c
    )

    model.load_state_dict(torch.load(load_path))
    return model


def test_private_model(model, test_loader):
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
    acc = 100 * correct / total
    return acc


def save_test_acc(acc, save_path):
    with open(save_path, "w") as f:
        f.write(str(acc))



if __name__ == "__main__":
    delta = 1e-5
    epsilons = [2.0, 4.0, 8.0]
    sigmas = [3.6, 2.0, 1.2]

    # learning_rates = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
    learning_rates = [0.1, 0.2, 0.4, 0.8, 1.6]
    # clipping_thresholds = [0.1, 0.4, 1.6, 6.4, 12.8]
    clipping_thresholds = [0.1, 0.4, 1.0, 1.6, 6.4]

    n_epoch = 75
    batch_size = 128

    train_loader, test_loader = read_cifar10(batch_size, "data/")

    for epsilon in epsilons:
        if epsilon == 4.0:
            for lr in learning_rates:
                for c in clipping_thresholds:
                    load_path = f"save/DPSGD/resnet20_{epsilon}e_{lr}lr_{c}c/model.pth"
                    model = load_private_model(load_path, train_loader, n_epoch, epsilon, delta, c)
                    acc = test_private_model(model, test_loader)
                    
                    print(f"epsilon: {epsilon}, lr: {lr}, c: {c}, test acc: {acc}")

                    save_path = f"save/DPSGD/resnet20_{epsilon}e_{lr}lr_{c}c/test_acc.txt"
                    save_test_acc(acc, save_path)

        

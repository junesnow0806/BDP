import torch

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

from resnet20 import resnet20
from train import train, test, save, load
from read_data import read_cifar10

import os

import warnings
warnings.simplefilter("ignore")

def train_with_epsilon(epsilon, delta, n_epoch, batch_size, lr, c):
    train_loader, _ = read_cifar10(batch_size, "data/")

    model = resnet20()
    model = ModuleValidator.fix(model)
    ModuleValidator.validate(model, strict=False)

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

    print(f"training model with epsilon: {epsilon}, lr: {lr}, c: {c}")

    model, training_loss, training_acc = train(model, optimizer, train_loader, n_epoch)
    return model, training_loss, training_acc


if __name__ == "__main__":
    delta = 1e-5
    epsilons = [2.0, 4.0, 8.0]
    sigmas = [3.6, 2.0, 1.2]

    learning_rates = [0.1, 0.2, 0.4, 0.8, 1.6]
    clipping_thresholds = [0.1, 0.4, 1.0, 1.6, 6.4]

    n_epoch = 100
    batch_size = 128

    for epsilon in epsilons:
        for lr in learning_rates:
            for c in clipping_thresholds:
                model, training_loss, training_acc = train_with_epsilon(epsilon, delta, n_epoch, batch_size, lr, c)

                save_dir = f"save/DPSGD/resnet20_{epsilon}e_{lr}lr_{c}c"
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                save(save_dir, model, training_loss, training_acc)



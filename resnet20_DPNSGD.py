import torch

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

from resnet20 import resnet20
from train import train, test, save, load
from read_data import read_cifar10
from Optimizer import NSGDOptimizer

import os

import warnings
warnings.simplefilter("ignore")

def train_with_epsilon(epsilon, delta, n_epoch, batch_size, lr, r, sigma):
    train_loader, _ = read_cifar10(batch_size, "data/")

    model = resnet20()
    model = ModuleValidator.fix(model)
    ModuleValidator.validate(model, strict=False)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    privacy_engine = PrivacyEngine()
    model, _, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=n_epoch,
        target_epsilon=epsilon,
        target_delta=delta,
        max_grad_norm=1.0
    )

    sample_rate = 1 / len(train_loader)
    expected_batch_size = int(len(train_loader.dataset) * sample_rate)
    optimizer = NSGDOptimizer(
        optimizer=optimizer,
        regularizer=r,
        noise_multiplier=sigma,
        expected_batch_size=expected_batch_size,
        max_grad_norm=1.0
    )

    print(f"training model with epsilon: {epsilon}, lr: {lr}, r: {r}")

    model, training_loss, training_acc = train(model, optimizer, train_loader, n_epoch)
    return model, training_loss, training_acc


if __name__ == "__main__":
    delta = 1e-5
    epsilons = [2.0, 4.0, 8.0]
    sigmas = [3.6, 2.0, 1.2]

    learning_rates = [0.1, 0.2, 0.4, 0.8, 1.6]
    regularizers = [0.0001, 0.001, 0.01, 0.1, 1.0]

    n_epoch = 100
    batch_size = 128

    for i, epsilon in enumerate(epsilons):
        sigma = sigmas[i]
        for lr in learning_rates:
            for r in regularizers:
                model, training_loss, training_acc = train_with_epsilon(epsilon, delta, n_epoch, batch_size, lr, r, sigma)

                save_dir = f"save/DPNSGD/resnet20_{epsilon}e_{lr}lr_{r}r"
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                save(save_dir, model, training_loss, training_acc)



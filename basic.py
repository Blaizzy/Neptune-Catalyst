
from catalyst import dl
from catalyst.contrib.datasets import MNIST
from catalyst.data.transforms import ToTensor
from collections import OrderedDict

from torch import nn, optim
from torch.utils.data import DataLoader
import os

my_hparams = {"lr": 0.03, "batch_size": 64}

model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), my_hparams["lr"])

loaders = OrderedDict(
    {
        "training": DataLoader(
            MNIST(os.getcwd(), train=True, download=True, transform=ToTensor()),
            batch_size=my_hparams["batch_size"],
        ),
        "validation": DataLoader(
            MNIST(os.getcwd(), train=False, download=True, transform=ToTensor()),
            batch_size=my_hparams["batch_size"],
        ),
    }
)

my_runner = dl.SupervisedRunner()


my_runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    loggers={
        "neptune": dl.NeptuneLogger(
            project="common/example-project-catalyst",
            tags=["datafest", "basic"],
            name="data-fest",
        )
    },
    loaders=loaders,
    num_epochs=5,
    callbacks=[
        dl.AccuracyCallback(
            input_key="logits",
            target_key="targets",
            topk_args=[1]
        ),
    ],
    hparams=my_hparams,
    logdir="./logs",
    valid_loader="validation",
    valid_metric="loss",
    minimize_valid_metric=True,
)

my_runner.log_artifact(
    path_to_artifact="./logs/checkpoints/best.pth",
    tag="best_model",
    scope="experiment"
)

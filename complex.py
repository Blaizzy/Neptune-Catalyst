# Adapted from the "Minimal Examples: CV - MNIST multistage finetuning":
# https://github.com/catalyst-team/catalyst#minimal-examples

import os

import matplotlib.pyplot as plt
from torch import nn, optim, squeeze
from torch.utils.data import DataLoader

from catalyst import dl, utils
from catalyst.contrib.datasets import MNIST
from catalyst.data.transforms import ToTensor

# define hyper-parameters
my_hparams = {
    "batch_size": 64,
    "linear": 256,
    "train_frozen_lr": 7e-3,
    "train_unfrozen_lr": 3e-2,
}


# create custom callback to log more metadata
class MyCallback(dl.Callback):
    """
    Log metadata at the different levels of the Catalyst run:

    * stage level:

        * on start: if "train_frozen" stage, log sample images
        * on end: if "train_unfrozen" stage, log mp4 file

    * epoch level:

        * on start: log audio file

    * loader level:

        * on end: log gif
    """

    def on_stage_start(self, runner: "IRunner") -> None:
        if runner.stage_key == "train_frozen":
            data = iter(runner.loaders["training"]).next()
            for image, label in zip(data[0], data[1]):
                image = squeeze(image)
                plt.figure(figsize=(10, 10))
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                plt.imshow(image, cmap=plt.cm.binary)
                runner.log_image(tag=str("class-{}".format(label)), image=plt.gcf())
                plt.close("all")

    def on_stage_end(self, runner: "IRunner") -> None:
        if runner.stage_key == "train_unfrozen":
            runner.log_artifact(
                path_to_artifact="./files/sac-rl.mp4", tag="video-stage-level", scope="stage"
            )

    def on_epoch_start(self, runner: "IRunner") -> None:
        runner.log_artifact(
            path_to_artifact="./files/elephant.wav", tag="audio-epoch-level", scope="epoch"
        )

    def on_loader_end(self, runner: "IRunner") -> None:
        if runner.loader_key == "validation":
            runner.log_artifact(
                path_to_artifact="./files/mean_action.gif", tag="gif-loader-level", scope="loader"
            )


class CustomRunner(dl.IRunner):
    def __init__(self, checkpoint_path, device, hparams):
        super().__init__()
        self._checkpoint_path = checkpoint_path
        self._device = device
        self._hparams = hparams

    def get_engine(self):
        return dl.DeviceEngine(self._device)

    # define NeptuneLogger
    def get_loggers(self):
        return {
            "neptune": dl.NeptuneLogger(
                project="common/example-project-catalyst",
                tags=["datafest", "complex"],
                name="data-fest",
            )
        }

    @property
    def stages(self):
        # suppose we have 2 stages:
        # 1st - with frozen encoder
        # 2nd with unfrozen whole network
        return ["train_frozen", "train_unfrozen"]

    def get_stage_len(self, stage: str) -> int:
        return 3

    def get_loaders(self, stage: str):
        loaders = {
            "training": DataLoader(
                MNIST(os.getcwd(), train=True, download=True, transform=ToTensor()),
                batch_size=my_hparams['batch_size']
            ),
            "validation": DataLoader(
                MNIST(os.getcwd(), train=False, download=True, transform=ToTensor()),
                batch_size=my_hparams['batch_size']
            ),
        }
        return loaders

    def get_model(self, stage: str):
        # the logic here is quite straightforward:
        # we create the model on the fist stage
        # and reuse it during next stages
        model = (
            self.model
            if self.model is not None
            else nn.Sequential(
                nn.Flatten(),
                nn.Linear(784, my_hparams['linear']),
                nn.ReLU(),
                nn.Linear(my_hparams['linear'], 10)
            )
        )
        if stage == "train_frozen":
            # 1st stage
            # freeze layer
            utils.set_requires_grad(model[1], False)
        else:
            # 2nd stage
            utils.set_requires_grad(model, True)
        return model

    def get_criterion(self, stage: str):
        return nn.CrossEntropyLoss()

    def get_optimizer(self, stage: str, model):
        if stage == "train_frozen":
            return optim.Adam(model.parameters(), lr=my_hparams['train_frozen_lr'])
        if stage == "train_unfrozen":
            return optim.SGD(model.parameters(), lr=my_hparams['train_unfrozen_lr'])

    def get_scheduler(self, stage: str, optimizer):
        return None

    def get_callbacks(self, stage: str):
        return {
            "criterion": dl.CriterionCallback(
                metric_key="loss", input_key="logits", target_key="targets"
            ),
            "optimizer": dl.OptimizerCallback(
                metric_key="loss"
            ),
            "accuracy": dl.AccuracyCallback(
                input_key="logits", target_key="targets", topk_args=[1]
            ),
            "checkpoint": dl.CheckpointCallback(
                logdir=self._checkpoint_path,
                loader_key="validation",
                metric_key="loss",
                minimize=True,
            ),
            # add MyCallback (defined before) to the callbacks list.
            "my_callback": MyCallback(order=1),
        }

    def handle_batch(self, batch):
        x, y = batch
        logits = self.model(x)

        self.batch = {
            "features": x,
            "targets": y,
            "logits": logits,
        }


# create runner instance
my_runner = CustomRunner("./checkpoints", "cpu", my_hparams)

# run the runner
my_runner.run()

# log best model to the run
my_runner.log_artifact(
    path_to_artifact="./checkpoints/best.pth", tag="best_model", scope="experiment"
)

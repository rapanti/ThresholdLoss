from torch import utils
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

import lightning.pytorch as pl

from data import MultiCropsTransform
from self_supervised import SimSiam


if __name__ == "__main__":
    model = SimSiam("resnet18_cifar", fix_pred_lr=True,)
    # # setup data
    dataset = CIFAR10("../../datasets/CIFAR10", download=True, transform=MultiCropsTransform(ToTensor(), 2))
    train_loader = utils.data.DataLoader(dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=8)

    trainer = pl.Trainer(
        max_epochs=10,
        limit_train_batches=10,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        default_root_dir="../exp")
    trainer.fit(model=model, train_dataloaders=train_loader, ckpt_path=None)

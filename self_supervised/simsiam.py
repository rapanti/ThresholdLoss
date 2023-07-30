import torch
import torch.nn as nn
import lightning.pytorch as pl

import models


class SimSiam(pl.LightningModule):

    def __init__(self,
                 base_encoder,
                 dim: int = 2048,
                 pred_dim: int = 512,
                 fix_pred_lr: bool = False,
                 learning_rate: float = 0.05,
                 momentum: float = 0.9,
                 weight_decay: float = 1e-4,
                 epochs: int = 100,
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = torchvision_ssl_encoder(base_encoder)
        encoder_out_dim = self.encoder(torch.rand(2, 3, 224, 224)).size(-1)

        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                       nn.BatchNorm1d(pred_dim),
                                       nn.ReLU(inplace=True),  # hidden layer
                                       nn.Linear(pred_dim, dim))  # output layer

        self.criterion = nn.CosineSimilarity(dim=-1)

    def forward(self, x):
        z = self.encoder(x)
        p = self.predictor(z)
        return p, z.detach()

    def training_step(self, batch, batch_idx):
        x1, x2 = batch[0]
        (p1, z1), (p2, z2) = self(x1), self(x2)
        loss = (-(self.criterion(p1, z2) + self.criterion(p2, z1)) * 0.5).mean()

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        if self.hparams.fix_pred_lr:
            params = [{'params': self.predictor.parameters(), 'lr': self.hparams.learning_rate},
                      {'params': self.encoder.parameters()}]
        else:
            params = self.parameters()

        optimizer = torch.optim.SGD(
            params,
            lr=self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            self.hparams.epochs,
        )
        return [optimizer], [scheduler]


class SiameseArm(nn.Module):

    def __init__(
        self,
        encoder: str = "resnet50",
        projector_out_dim: int = 2048,
    ) -> None:
        super().__init__()

        self.encoder = torchvision_ssl_encoder(encoder)
        encoder_out_dim = self.encoder(torch.rand(2, 3, 224, 224)).size(-1)

        self.encoder.fc = ProjectorMLP(encoder_out_dim, encoder_out_dim, projector_out_dim)

    def forward(self, x):
        return self.encoder(x)


class ProjectorMLP(nn.Module):

    def __init__(self, input_dim: int = 2048, hidden_dim: int = 2048, output_dim: int = 2048) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim, bias=False),
            nn.BatchNorm1d(output_dim, affine=False)
        )

    def forward(self, x):
        return self.model(x)


class MLP(nn.Module):

    def __init__(self, input_dim: int = 2048, hidden_dim: int = 512, output_dim: int = 2048) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim, bias=True),
        )

    def forward(self, x):
        return self.model(x)


def torchvision_ssl_encoder(name):
    model = getattr(models, name)()
    model.fc = nn.Identity()
    return model

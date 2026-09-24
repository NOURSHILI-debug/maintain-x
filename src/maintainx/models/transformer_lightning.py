"""LightningModule wrapping RULTransformer: training/val loop, optimizer, logging."""

import torch
import torch.nn.functional as F
import lightning as L

from maintainx.models.transformer import RULTransformer


class RULTransformerModule(L.LightningModule):
    def __init__(self, n_features, d_model=64, n_heads=4, n_layers=2,
                 d_ff=128, dropout=0.1, lr=1e-3, rul_cap=125):
        super().__init__()
        self.save_hyperparameters()  # logs these to MLflow automatically
        self.model = RULTransformer(
            n_features=n_features, d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, d_ff=d_ff, dropout=dropout,
        )
        self.rul_cap = rul_cap

    def forward(self, x):
        return self.model(x)
    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")
    def _shared_step(self, batch, stage):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        rmse = torch.sqrt(loss)
        mae = F.l1_loss(y_hat, y)
        self.log(f"{stage}_loss", loss, prog_bar=(stage == "train"))
        self.log(f"{stage}_rmse", rmse, prog_bar=True)
        self.log(f"{stage}_mae", mae)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
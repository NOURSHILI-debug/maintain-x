"""Train the RUL transformer and log it to the rul-fd001 MLflow experiment."""

import lightning as L
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from maintainx.models.transformer_lightning import RULTransformerModule
from maintainx.models.transformer_data import RULWindowDataModule
from maintainx.tracking.mlflow_utils import setup_mlflow


def main():
    uri = setup_mlflow("rul-fd001")

    dm = RULWindowDataModule(window=30, stride=5, batch_size=64)
    dm.setup()

    model = RULTransformerModule(
        n_features=dm.n_features, d_model=64, n_heads=4, n_layers=2,
        d_ff=128, dropout=0.1, lr=1e-3,
    )

    mlf_logger = MLFlowLogger(
        experiment_name="rul-fd001",
        tracking_uri=uri,
        run_name="transformer",
    )

    trainer = L.Trainer(
        max_epochs=100,
        logger=mlf_logger,
        callbacks=[
            EarlyStopping(monitor="val_rmse", patience=10, mode="min"),
            ModelCheckpoint(monitor="val_rmse", mode="min", save_top_k=1),
        ],
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()
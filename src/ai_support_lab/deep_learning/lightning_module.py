from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import nn

from ai_support_lab.deep_learning.dataset import LABELS, prepare_data
from ai_support_lab.deep_learning.model import TicketTextClassifier
from ai_support_lab.deep_learning.training import save_inference_checkpoint


class TicketLightningModule(L.LightningModule):
    def __init__(self, vocabulary_size: int, classes: int, learning_rate: float = 0.01) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.model = TicketTextClassifier(vocabulary_size, classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.model(tokens)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        tokens, targets = batch
        loss = self.criterion(self(tokens), targets)
        # В automatic optimization Lightning выполняет zero_grad/backward/step.
        # Здесь остаётся содержательная часть шага, а не второй скрытый training loop.
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=len(targets))
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        tokens, targets = batch
        logits = self(tokens)
        self.log("val_loss", self.criterion(logits, targets), batch_size=len(targets))
        self.log(
            "val_accuracy",
            (logits.argmax(dim=1) == targets).float().mean(),
            batch_size=len(targets),
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


def train(path: Path, output: Path, epochs: int = 8, accelerator: str = "cpu") -> str:
    L.seed_everything(42, workers=True)
    torch.set_num_threads(2)
    data = prepare_data(path)
    module = TicketLightningModule(len(data.vocabulary), len(LABELS))
    checkpoint = ModelCheckpoint(
        dirpath=output.parent / "lightning", monitor="val_loss", mode="min", save_top_k=1
    )
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=1,
        logger=False,
        callbacks=[checkpoint],
        gradient_clip_val=1.0,
        enable_progress_bar=False,
    )
    trainer.fit(module, data.train, data.validation)
    # Экспортируем ту же nn.Module в тот же inference format, что и manual loop.
    # Lightning checkpoint дополнительно содержит состояние Trainer/optimizer.
    best = TicketLightningModule.load_from_checkpoint(
        checkpoint.best_model_path, map_location="cpu"
    )
    save_inference_checkpoint(best.model, data.vocabulary, output)
    return checkpoint.best_model_path

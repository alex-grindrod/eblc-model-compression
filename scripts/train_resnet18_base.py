import sys
sys.path.append(".")
import pytorch_lightning as pl
from src import data_loaders
from src.models import resnet_18
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import WandbLogger
import os

EPOCHS = 20

def train():
    wandb_logger = WandbLogger(project="Resnet18_190", entity="agrindro")
    model = resnet_18.ResNetLightning()

    train_loader, val_loader = data_loaders.cifar10_loader(train=True, num_workers=1)
    test_loader = data_loaders.cifar10_loader(train=False, num_workers=1)

    save_path = os.path.join(os.getcwd(), "models", "resnet18_base_best")

    callbacks = [ModelCheckpoint(filename=save_path, monitor='val_accuracy', mode='max', save_top_k=1, verbose=True, every_n_epochs=1),
                RichProgressBar()
                ]

    trainer = pl.Trainer(logger=wandb_logger, callbacks=callbacks, max_epochs=EPOCHS)
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    train()
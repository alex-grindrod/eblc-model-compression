
import sys
sys.path.append('.')

import pytorch_lightning as pl
import torch
from src import data_loaders
from src.models import resnet_18
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import WandbLogger

def eval(model=None, ckpt_file=None, device=torch.device('cpu')):
    if not model and not ckpt_file:
        print("Provide either model or ckpt file")
        return
    elif ckpt_file:
        model = resnet_18.ResNetLightning()
        ckpt = torch.load(ckpt_file, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
    
    # wandb_logger = WandbLogger(project="Resnet18_190", entity="agrindro")
    test_loader = data_loaders.cifar10_loader(train=False, num_workers=4)
    callbacks = [ModelCheckpoint(filename="/kaggle/working/models/best", monitor='val_accuracy', mode='max', save_top_k=1, verbose=True, every_n_epochs=1),
                RichProgressBar()
                ]

    # trainer = pl.Trainer(logger=wandb_logger, callbacks=callbacks, max_epochs=10)
    trainer = pl.Trainer(callbacks=callbacks, max_epochs=10)
    trials = trainer.test(model, dataloaders=test_loader)
    print(f"Final Test Accuracy: {trials[0]['test_acc']:.4f}")

if __name__ == "__main__":
    eval(ckpt_file="models/resnet18_base_best-v1.ckpt", device=torch.device('gpu'))
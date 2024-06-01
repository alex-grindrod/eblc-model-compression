import torch
import torch.nn.functional as F
from torchvision import models
import pytorch_lightning as pl

#Built for CIFAR 10
class ResNetLightning(pl.LightningModule):
    def __init__(self, num_classes=10):
        super(ResNetLightning, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # Example: Using ResNet18
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)  # Adjust for 10 classes (e.g., CIFAR-10)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        accuracy = torch.sum(preds == y).float() / len(y)
        self.log('val_loss', loss)
        self.log('val_accuracy', accuracy)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        accuracy = torch.sum(preds == y).float() / len(y)
        # self.log('test_loss', loss)
        # self.log('test_acc', accuracy)
        return {'test_loss': loss, 'test_acc': accuracy}

    
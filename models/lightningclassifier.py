import numpy as np
from models.pretrained_models import get_model_output
import pytorch_lightning as pl
from torch import nn
from common_blocks.losses import LabelSmoothingCrossEntropy
import torch
from common_blocks.utils import qwk_metric
from pytorch_lightning.core.memory import ModelSummary
from pytorch_lightning import _logger as log
from torch.optim.lr_scheduler import ReduceLROnPlateau

class LightningCanserClassifier(pl.LightningModule):
    def __init__(self, config):
        super(LightningCanserClassifier, self).__init__()
        self.config = config
        self.model = get_model_output(**config['model_params'])  # CustomSEResNeXt(config['model_params'])

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        return self.model.forward(x)

    def get_loss(self, y_preds, labels):
        if self.config['training']['loss'] == 'CrossEntropyLoss':
            loss_func = nn.CrossEntropyLoss()
            return loss_func(y_preds, labels.long())
        elif self.config['training']['loss'] == 'LabelSmoothingCrossEntropy':
            loss_func = LabelSmoothingCrossEntropy()
            return loss_func(y_preds, labels.long())
        else:
            raise NotImplementedError("This loss {} isn't implemented".format(config['training']['loss']))

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_preds = self.forward(x)
        loss = self.get_loss(y_preds, y)
        with torch.no_grad():
            metric = qwk_metric(y_preds, y)

        logs = {'train_loss': loss, 'train_metric': metric}
        progress_bar = {'train_metric': metric}
        return {'loss': loss, 'metric': metric, 'log': logs, "progress_bar": progress_bar}

    def training_epoch_end(self, outputs):

        avg_loss_train = torch.stack([x['loss'] for x in outputs]).mean()
        avg_metric_train = np.stack([x['metric'] for x in outputs]).mean()

        tensorboard_logs = {'avg_train_loss': avg_loss_train, 'avg_train_metric': avg_metric_train}
        print('\ntrain', 'avg_train_metric', avg_metric_train)
        return {'avg_train_loss': avg_loss_train, 'avg_train_metric': avg_metric_train, 'log': tensorboard_logs}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        with torch.no_grad():
            y_preds = self.forward(x)
        loss = self.get_loss(y_preds, y)
        metric = qwk_metric(y_preds, y)
        return {'val_loss': loss, 'val_metric': metric}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_metric = np.stack([x['val_metric'] for x in outputs]).mean()

        tensorboard_logs = {'avg_val_loss': avg_loss, 'avg_val_metric': avg_metric}
        print('\nval', tensorboard_logs, 'avg_val_metric', avg_metric)
        return {'avg_val_loss': avg_loss, 'avg_val_metric': avg_metric, 'log': tensorboard_logs,
                "progress_bar": tensorboard_logs}

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure):
        if self.trainer.global_step < self.config['training']['warmup_steps']:
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.config['training']['warmup_steps'])
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.config['training']['optimizer']['kwargs']['lr']
        optimizer.step()
        optimizer.zero_grad()

    def prepare_data(self):
        pass

    def summarize(self, mode: str) -> None:
        if self.config['model_params']['show_model_summary']:
            model_summary = ModelSummary(self, mode=mode)
            log.info('\n' + model_summary.__str__())

    def configure_optimizers(self):
        if self.config['training']['optimizer']['name'] == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), **self.config['training']['optimizer']['kwargs'])
        else:
            NotImplementedError("This optimizer {} isn't implemented".format(self.config['training']['optimizer']['name']))

        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, **self.config['training']['scheduler']['ReduceLROnPlateau']),
            **self.config['training']['scheduler']['kwargs']
        }
        return [optimizer], [scheduler]

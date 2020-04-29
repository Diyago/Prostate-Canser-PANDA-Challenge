import codecs
import os
import warnings

import numpy as np
import pytorch_lightning as pl
import torch
from poyo import parse_string
from pytorch_lightning import _logger as log
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.core.memory import ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from common_blocks.datasets import TrainDataset
from common_blocks.losses import LabelSmoothingCrossEntropy
from common_blocks.logger import init_logger
from common_blocks.transforms import get_transforms
from common_blocks.utils import seed_torch, create_folds, qwk_metric
from models.seresnext import CustomSEResNeXt
from models.pretrained_models import get_model_output

with codecs.open("config/config_regression.yml", encoding="utf-8") as ymlfile:
    config_yaml = ymlfile.read()
    config = parse_string(config_yaml)

LOGGER = init_logger(config['logger_path']['main_logger'])
warnings.filterwarnings("ignore", category=RuntimeWarning)


class LightningCanserClassifier(pl.LightningModule):
    def __init__(self, config):
        super(LightningCanserClassifier, self).__init__()

        self.config = config
        self.model = get_model_output('efficientnet_b3b', num_outputs=1)

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        return self.model.forward(x)

    def get_loss(self, y_preds, labels):
        loss_func = nn.MSELoss()
        return loss_func(y_preds, labels.float())
     #   else:
     #      raise NotImplementedError("This loss {} isn't implemented".format(config['training']['loss']))

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_preds = self.forward(x)
        loss = self.get_loss(y_preds, y)
        with torch.no_grad():
            metric = qwk_metric(y_preds, y)
        logs = {'train_loss': loss, 'train_metric': metric}
        return {'loss': loss, 'metric': metric, 'log': logs}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        with torch.no_grad():
            y_preds = self.forward(x)

        loss = self.get_loss(y_preds, y)
        metric = qwk_metric(y_preds, y)
        return {'val_loss': loss, 'val_metric': metric}

    def training_epoch_end(self, outputs):

        avg_loss_train = torch.stack([x['loss'] for x in outputs]).mean()
        avg_metric_train = np.stack([x['metric'] for x in outputs]).mean()

        tensorboard_logs = {'avg_train_loss': avg_loss_train, 'avg_train_metric': avg_metric_train}
        return {'avg_train_loss': avg_loss_train, 'avg_train_metric': avg_metric_train, 'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_metric = np.stack([x['val_metric'] for x in outputs]).mean()

        tensorboard_logs = {'avg_val_loss': avg_loss, 'avg_val_metric': avg_metric}
        return {'avg_val_loss': avg_loss, 'avg_val_metric': avg_metric, 'log': tensorboard_logs}

    def prepare_data(self):
        pass

    def configure_optimizers(self):
        if config['training']['optimizer']['name'] == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), **config['training']['optimizer']['kwargs'])
        else:
            NotImplementedError("This optimizer {} isn't implemented".format(config['training']['optimizer']['name']))

        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, **config['training']['scheduler']['ReduceLROnPlateau']),
            **config['training']['scheduler']['kwargs']
        }
        return [optimizer], [scheduler]

    # def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure):
    #     if self.trainer.global_step < config['training']['warmup_steps']:
    #         lr_scale = min(1., float(self.trainer.global_step + 1) / config['training']['warmup_steps'])
    #         for pg in optimizer.param_groups:
    #             pg['lr'] = lr_scale * self.hparams.learning_rate
    #
    #     optimizer.step()
    #     optimizer.zero_grad()

    def summarize(self, mode: str) -> None:
        if config['model_params']['show_model_summary']:
            model_summary = ModelSummary(self, mode=mode)
            log.info('\n' + model_summary.__str__())


if __name__ == '__main__':
    seed_torch(seed=config['total_seed'])
    folds = create_folds(config['validation'])


    for fold in range(config['validation']['nfolds']):
        trn_idx = folds[folds['fold'] != fold].index
        val_idx = folds[folds['fold'] == fold].index

        train_dataset = TrainDataset(folds.loc[trn_idx].reset_index(drop=True),
                                     config['Train']['Dataset'],
                                     transform=get_transforms(data='train'))
        valid_dataset = TrainDataset(folds.loc[val_idx].reset_index(drop=True),
                                     config['Val']['Dataset'],
                                     transform=get_transforms(data='valid'))

        train_loader = DataLoader(train_dataset, **config['Train']['loader'])
        valid_loader = DataLoader(valid_dataset, **config['Val']['loader'])

        tb_logger = TensorBoardLogger(save_dir=config['logger_path']['lightning_logger'],
                                      name=config['model_params']['model']['name'], version=f'fold_{fold + 1}')
        os.makedirs('{}/{}'.format(config['logger_path']['lightning_logger'], config['model_params']['model']['name']),
                    exist_ok=True)

        checkpoint_callback = ModelCheckpoint(
            filepath=tb_logger.log_dir + config['training']['ModelCheckpoint']['path'],
            **config['training']['ModelCheckpoint']['kwargs'])
        early_stop_callback = EarlyStopping(**config['training']['early_stop_callback'])
        model = LightningCanserClassifier(config)

        trainer = pl.Trainer(logger=tb_logger, early_stop_callback=early_stop_callback,
                             checkpoint_callback=checkpoint_callback, **config['training']['Trainer'])
        trainer.fit(model, train_dataloader=train_loader, val_dataloaders=valid_loader)

import numpy as np
import torch
from torchvision.utils import make_grid
from torchvision import transforms
from base import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
        Modify __init__() if you have additional arguments to pass.
    """
    def __init__(self, model, loss, metrics, data_loader, optimizer, epochs,
                 save_dir, save_freq, resume, device, verbosity, training_name='',
                 valid_data_loader=None, train_logger=None, writer=None, lr_scheduler=None, monitor='loss', monitor_mode='min'):
        super(Trainer, self).__init__(model, loss, metrics, data_loader, valid_data_loader, optimizer, epochs,
                                      save_dir, save_freq, resume, verbosity, training_name,
                                      device, train_logger, writer, monitor, monitor_mode)
        self.scheduler = lr_scheduler

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log
        """
        self.model.train()
        self.model.to(self.device)

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, (data, label) in enumerate(self.data_loader):

            real_label = 1
            fake_label = 0
            
            data = [d.to(self.device) for d in data]

            # train D with real data
            self.d_optimizer.zero_grad()
            # output, fake_x = self.model(z, data)
            _, cond = self.model.G(label)
            output = self.model.D(data[0], cond)
            errD_real = self.loss(output, real_label)
            errD_real.backward(retain_graph=True)

            # train D with fake data
            # label.fill_(fake_label)
            fake_x, _ = self.model.G(label)

            output = self.model.D(fake_x, cond)
            errD_fake = self.loss(output, fake_label)
            errD_fake.backward(retain_graph=True)

            self.d_optimizer.step()

            self.g_optimizer.zero_grad()
            # train G
            # label.fill_(real_label)
            # output = self.model.D(fake_x)
            errG = self.loss(output, real_label)
            errG.backward()

            loss_D = errD_fake.item() + errD_real.item()
            loss_G = errG.item()
            loss = loss_G + loss_D

            self.g_optimizer.step()

            self.train_iter += 1
            self.writer.add_scalar(f'{self.training_name}/Train/D_loss', loss_D, self.train_iter)
            self.writer.add_scalar(f'{self.training_name}/Train/G_loss', loss_G, self.train_iter)
        
            if self.train_iter % 50 == 0:
                # self.writer.add_image('image/orig', data, self.train_iter)
                self.writer.add_image('image/generated', make_grid(fake_x, normalize=True), self.train_iter)

            total_loss += loss
            log_step = int(np.sqrt(self.batch_size))
            if self.verbosity >= 2 and batch_idx % log_step == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.data_loader) * len(data),
                    100.0 * batch_idx / len(self.data_loader), loss))

        avg_loss = total_loss / len(self.data_loader)
        avg_metrics = (total_metrics / len(self.data_loader)).tolist()
        log = {'loss': avg_loss, 'metrics': avg_metrics}

        if self.valid:
            val_log = self._valid_epoch()
            log = {**log, **val_log}

        return log

    def _valid_epoch(self):
        """
        Validate after training an epoch

        :return: A log that contains information about validation
        """
        self.model.eval()
        with torch.no_grad():
            total_val_loss = 0
            total_val_metrics = np.zeros(len(self.metrics))
            for batch_idx, (data, _) in enumerate(self.valid_data_loader):
                data = data.to(self.device)
                batch_size = data.shape[0]
                z = torch.randn((batch_size, 100, 1, 1), device=self.device)

                output, fake_x = self.model(z, data)
                loss = self.loss(output[:batch_size], output[batch_size:])
                total_val_loss += loss.item()

                self.valid_iter += 1
                self.writer.add_scalar(f'{self.training_name}/Valid/loss', loss.item(), self.valid_iter)
                for i, metric in enumerate(self.metrics):
                    score = metric(output, target)
                    total_val_metrics[i] += score
                    self.writer.add_scalar(f'{self.training_name}/Valid/{metric.__name__}', score, self.valid_iter)

            avg_val_loss = total_val_loss / len(self.valid_data_loader)
            if self.scheduler is not None:
                self.scheduler.step(avg_val_loss)
            avg_val_metrics = (total_val_metrics / len(self.valid_data_loader)).tolist()
        return {'val_loss': avg_val_loss, 'val_metrics': avg_val_metrics}

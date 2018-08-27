import numpy as np
import torch
from torch.nn.utils.rnn import pad_packed_sequence
from torchvision.utils import make_grid
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

        word2index = self.data_loader.dataset.preprocessed.word2index
        self.index2word = {v:k for k, v in word2index.items()}
        self.index2word[1] = ''
        self.index2word[0] = ''

    def decode_sentence(self, index_tensor):
        index_tensor = index_tensor.cpu()
        batch = torch.unbind(pad_packed_sequence(index_tensor)[0], 1)
        lengths = pad_packed_sequence(index_tensor)[1]
        sentences = []
        for sent, length in zip(batch, lengths):
            sentence = ' '.join([self.index2word[idx.item()] for idx in sent[:length]])
            sentences.append(sentence)
        return sentences        

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
            fake_x_0, fake_x_1, fake_x_2, cond, mu, std = self.model.G(label)
            output_0 = self.model.D(data[0], cond)
            output_1 = self.model.D(data[1], cond)
            output_2 = self.model.D(data[2], cond)

            errD_real = self.loss(output_0, real_label) + self.loss(output_1, real_label) + self.loss(output_2, real_label)
            # errD_real = self.loss(output_0, real_label) + self.loss(output_1, real_label) 
            errD_real.backward(retain_graph=True)            

            # train D with fake data
            output_0 = self.model.D(fake_x_0, cond)
            output_1 = self.model.D(fake_x_1, cond)
            output_2 = self.model.D(fake_x_2, cond)

            # errD_fake = self.loss(output_0, fake_label) + self.loss(output_1, fake_label) 
            errD_fake = self.loss(output_0, fake_label) + self.loss(output_1, fake_label) + self.loss(output_2, fake_label)
            errD_fake.backward(retain_graph=True)
            self.d_optimizer.step()

            # train G
            self.g_optimizer.zero_grad()
            # errG = self.loss(output_0, real_label, mu, std) + self.loss(output_1, real_label, mu, std) 
            errG = self.loss(output_0, real_label, mu, std) + self.loss(output_1, real_label, mu, std) + self.loss(output_2, real_label, mu, std)
            errG.backward()
            self.g_optimizer.step()
            
            loss_D = errD_fake.item() + errD_real.item()
            loss_G = errG.item()
            loss = loss_G + loss_D


            self.train_iter += 1
            self.writer.add_scalar(f'{self.training_name}/Train/D_loss', loss_D, self.train_iter)
            self.writer.add_scalar(f'{self.training_name}/Train/G_loss', loss_G, self.train_iter)
            if self.train_iter % 20 == 0:
                # self.writer.add_image('image/original', make_grid(data[0], normalize=True), self.train_iter)
                self.writer.add_image('image/generated_0', make_grid(fake_x_0, normalize=True), self.train_iter)
                self.writer.add_image('image/generated_1', make_grid(fake_x_1, normalize=True), self.train_iter)
                self.writer.add_image('image/generated_2', make_grid(fake_x_2, normalize=True), self.train_iter)
                self.writer.add_text('text', '\n\n'.join(self.decode_sentence(label)), self.train_iter) # this is not working yet
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

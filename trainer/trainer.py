import numpy as np
import torch
from torch.nn.utils.rnn import pad_packed_sequence
from torchvision.utils import make_grid
from base import BaseTrainer
from torchvision import transforms
from model.model import Matching_Score_word, Matching_Score_sent

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
        self.loss = loss['gan']
        self.kld = loss['kld']
        self.damsm_loss = loss['damsm']

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
        return '\n\n'.join(sentences)

    def step_optims(self, names):
        if isinstance(names, list):
            for k in names:
                self.optimizer[k].step()
        else:
            self.optimizer[names].step()
    
    def init_optims(self, names):
        if isinstance(names, list):
            for k in names:
                self.optimizer[k].zero_grad()
        else:
            self.optimizer[names].zero_grad()

    def reshape_output(self, image_batch):
        transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize(80), transforms.ToTensor()])
        result = []
        for img in torch.unbind(image_batch, dim=0):
            result.append(transform(img))
        return torch.cat(result, dim=0)
        
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
        for batch_idx, (data, text) in enumerate(self.data_loader):
            real_label = 1
            fake_label = 0
            
            data = [d.to(self.device) for d in data]

            text_embedded, z_input, cond, mu, std, sen_feature = self.model.prepare_inputs(text)
            
            # train F_ca according to mu, std
            self.init_optims('F_ca')
            loss_F_ca = self.kld(mu, std)
            loss_F_ca.backward(retain_graph=True)
            self.step_optims('F_ca')

            # train D with real images
            self.init_optims('D_0')
            output_0 = self.model.D_0(data[0], cond.detach())

            errD_real_0 = self.loss(output_0, real_label)
            errD_real_0.backward(retain_graph=True)            
            self.step_optims('D_0')

            # Stage 1
            #
            h_0, fake_x_0 = self.model.F_0(z_input)
            # train D_0 with fake data
            self.init_optims('D_0')
            score_fake_0 = self.model.D_0(fake_x_0, cond)
            errD_fake_0 = self.loss(score_fake_0, fake_label)
            errD_fake_0.backward(retain_graph=True)
            self.step_optims('D_0')
            
            # train G_0 with fake data
            update_targets = ['F_0', 'Text_encoder']
            self.init_optims(update_targets)
            errG_0 = self.loss(score_fake_0, real_label)
            errG_0.backward(retain_graph=True)
            self.step_optims(update_targets)


            self.train_iter += 1
            if epoch <= 0:
                loss_D = errD_fake_0.item() + errD_real_0.item()
                loss_G = errG_0.item()
                loss = loss_G + loss_D       

                self.writer.add_scalar(f'{self.training_name}/Train/global/D_loss_real', errD_real_0.item(), self.train_iter)
                self.writer.add_scalar(f'{self.training_name}/Train/global/F_ca_loss', loss_F_ca.item(), self.train_iter)
                self.writer.add_scalar(f'{self.training_name}/Train/stage0/D_loss_fake', errD_fake_0.item(), self.train_iter)
                self.writer.add_scalar(f'{self.training_name}/Train/stage0/G_loss', errG_0.item(), self.train_iter)
                if self.train_iter % 20 == 0:
                    self.writer.add_image('image/generated_0', make_grid(fake_x_0[:16], normalize=True, nrow=4), self.train_iter)

            else:
                update_targets = ['D_1', 'D_2']
                self.init_optims(update_targets)
                output_1 = self.model.D_1(data[1], cond.detach())
                output_2 = self.model.D_2(data[2], cond.detach())
                errD_real = self.loss(output_1, real_label) + self.loss(output_2, real_label)
                errD_real.backward(retain_graph=True)
                self.step_optims(update_targets)
                #
                # Stage 2
                #
                c_0 = self.model.F_1_attn(text_embedded, h_0.detach()) # detach for isolation of graph from stage 1
                h_1, fake_x_1 = self.model.F_1(c_0, h_0.detach()) 
                
                # train D_1 with fake data
                self.init_optims('D_1')
                score_fake_1 = self.model.D_1(fake_x_1, cond)
                errD_fake_1 = self.loss(score_fake_1, fake_label)
                errD_fake_1.backward(retain_graph=True)
                self.step_optims('D_1')
                
                # train G_1 with fake data
                update_targets = ['F_1_attn', 'F_1', 'Text_encoder']
                self.init_optims(update_targets)
                errG_1 = self.loss(score_fake_1, real_label)
                errG_1.backward(retain_graph=True)
                self.step_optims(update_targets)

                #
                # Stage 3
                #
                c_1 = self.model.F_2_attn(text_embedded, h_1.detach()) # detach for isolation of graph from stage 1
                h_2, fake_x_2 = self.model.F_1(c_1, h_1.detach()) 

                # train D_2 with fake data
                self.init_optims('D_2')
                score_fake_2 = self.model.D_2(fake_x_2, cond)
                errD_fake_2 = self.loss(score_fake_2, fake_label)
                errD_fake_2.backward(retain_graph=True)
                self.step_optims('D_2')
                
                # train G_2 with fake data
                update_targets = ['F_2_attn', 'F_2', 'Text_encoder']
                self.init_optims(update_targets)
                errG_2 = self.loss(score_fake_2, real_label)
                errG_2.backward(retain_graph=True)
                self.step_optims(update_targets)

                matching_score_word = Matching_Score_word(5, 5, 10)
                matching_score_sent = Matching_Score_sent(10)
                
                update_targets = ['Image_encoder', 'Text_encoder']
                reshaped_output = self.reshape_output(fake_x_2)
                local_feature, global_feature = self.model.image_encoder(reshaped_output)
                b, c, _, _ = local_feature.shape
                word_score_1, word_score_2 = matching_score_word(text_embedded, local_feature.view(b, c, -1))
                sent_score_1, sent_score_2 = matching_score_sent(sen_feature, global_feature)
                loss_damsm = self.damsm_loss(word_score_1, 10) + self.damsm_loss(word_score_2, 10) + self.damsm_loss(sent_score_1, 10) + self.damsm_loss(sent_score_2, 10)
                loss_damsm.backward(retain_graph=True)
                

                loss_D = errD_fake_0.item() + errD_fake_1.item() + errD_fake_2.item() + errD_real_0.item() + errD_real.item()
                loss_G = errG_0.item() + errG_1.item() + errG_2.item()
                loss = loss_G + loss_D
                

                self.writer.add_scalar(f'{self.training_name}/Train/stage1/D_loss_fake', errD_fake_1.item(), self.train_iter)
                self.writer.add_scalar(f'{self.training_name}/Train/stage2/D_loss_fake', errD_fake_2.item(), self.train_iter)
                self.writer.add_scalar(f'{self.training_name}/Train/stage2/D_loss_real', errD_real.item(), self.train_iter)

                self.writer.add_scalar(f'{self.training_name}/Train/stage1/G_loss', errG_1.item(), self.train_iter)
                self.writer.add_scalar(f'{self.training_name}/Train/stage2/G_loss', errG_2.item(), self.train_iter)
                
                self.writer.add_scalar(f'{self.training_name}/Train/stage3/damsm_loss', loss_damsm.item(), self.train_iter)

                if self.train_iter % 20 == 0:
                    self.writer.add_image('image/generated_0', make_grid(fake_x_0, normalize=True, nrow=4), self.train_iter)
                    self.writer.add_image('image/generated_1', make_grid(fake_x_1, normalize=True, nrow=4), self.train_iter)
                    self.writer.add_image('image/generated_2', make_grid(fake_x_2, normalize=True, nrow=4), self.train_iter)
                    self.writer.add_text('text', self.decode_sentence(text), self.train_iter)

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

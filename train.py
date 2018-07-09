import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model.model import DC_GAN
from model.loss import gan_loss
# from model.metric import accuracy
from data_loader import CocoDataLoader, CubDataLoader
from trainer import Trainer
from logger import Logger
from tensorboardX import SummaryWriter

logging.basicConfig(level=logging.INFO, format='')
writer = SummaryWriter('saved/runs')


parser = argparse.ArgumentParser(description='PyTorch Template')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    help='mini-batch size (default: 32)')
parser.add_argument('-e', '--epochs', default=32, type=int,
                    help='number of total epochs (default: 32)')
parser.add_argument('--lr', default=0.0002, type=float,
                    help='learning rate (default: 0.0002)')
parser.add_argument('--wd', default=0.0, type=float,
                    help='weight decay (default: 0.0)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--verbosity', default=2, type=int,
                    help='verbosity, 0: quiet, 1: per epoch, 2: complete (default: 2)')
parser.add_argument('--save-dir', default='saved', type=str,
                    help='directory of saved model (default: saved)')
parser.add_argument('--save-freq', default=1, type=int,
                    help='training checkpoint frequency (default: 1)')
parser.add_argument('--data-dir', default='datasets', type=str,
                    help='directory of training/testing data (default: datasets)')
parser.add_argument('--valid-batch-size', default=1000, type=int,
                    help='mini-batch size (default: 1000)')
parser.add_argument('--validation-split', default=0.0, type=float,
                    help='ratio of split validation data, [0.0, 1.0) (default: 0.1)')
parser.add_argument('--validation-fold', default=0, type=int,
                    help='select part of data to be used as validation set (default: 0)')
parser.add_argument('--no-cuda', action="store_true",
                    help='use CPU instead of GPU')


def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    # Model
    # G = DC_Generator(100, n_size=2)
    # D = DC_Discriminator(100, n_size=2)
    model = DC_GAN(100, 3, n_size=4)
    # model = (G, D)
    
    # print('generator: \n', G)
    # print('discriminator: \n', D)
    model.summary()

    # A logger to store training process information
    train_logger = Logger()

    # Specifying loss function, metric(s), and optimizer
    loss = gan_loss
    metrics = []
    g_optimizer = optim.Adam(model.G.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=True, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(model.D.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=True, betas=(0.5, 0.999))

    # Data loader and validation split
    data_loader = CubDataLoader('../data/birds', args.batch_size, args.valid_batch_size, args.validation_split, args.validation_fold, shuffle=True, num_workers=4)
    # data_loader = CocoDataLoader('../cocoapi', args.batch_size, args.valid_batch_size, args.validation_split, args.validation_fold, shuffle=True, num_workers=4)
    valid_data_loader = data_loader.get_valid_loader()

    # An identifier for this training session
    training_name = type(model).__name__

    # Trainer instance
    trainer = Trainer(model, loss, metrics,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      optimizer=(g_optimizer, d_optimizer),
                      epochs=args.epochs,
                      train_logger=train_logger,
                      writer=writer,
                      save_dir=args.save_dir,
                      save_freq=args.save_freq,
                      resume=args.resume,
                      verbosity=args.verbosity,
                      training_name=training_name,
                      device=device,
                      monitor='loss',
                      monitor_mode='max')

    # Start training!
    trainer.train()

    # See training history
    print(train_logger)


if __name__ == '__main__':
    main(parser.parse_args())

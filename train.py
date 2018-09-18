import argparse
import logging
import torch
import torch.optim as optim
# from torch.optim.lr_scheduler import ReduceLROnPlateau
from model.model import AttnGAN
from model.loss import gan_loss, kld_loss, DAMSM_loss
# from model.metric import accuracy
from data_loader import CocoDataLoader, CubDataLoader
from trainer import Trainer
from logger import Logger
from tensorboardX import SummaryWriter
# import torch.multiprocessing as mp
from datetime import datetime
import torch.nn as nn 

logging.basicConfig(level=logging.INFO, format='')

def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-b', '--batch-size', default=10, type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument('-e', '--epochs', default=32, type=int,
                        help='number of total epochs (default: 32)')
    parser.add_argument('--lr', default=0.0002, type=float,
                        help='learning rate (default: 0.0002)')
    parser.add_argument('--wd', default=0.0, type=float,
                        help='weight decay (default: 0.0)')
    parser.add_argument('--resume', default=None, type=str,
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
    parser.add_argument('--timestamp', default=datetime.now().strftime("%y%m%d%H%M%S"), type=str)
    parser.add_argument('--log-dir', default='saved/runs/', type=str)
    parser.add_argument('--in-ch', default=3, type=int)
    parser.add_argument('--vocab-size', default=4795, type=int)
    parser.add_argument('--latent-size', default=128, type=int)
    parser.add_argument('--dropout', default=0.2, type= float)
    parser.add_argument('--embedding-size', default=128, type=int)
    parser.add_argument('--hidden-size', default=64, type=int)

    args = parser.parse_args()
    config_list = [args.batch_size, args.lr, args.in_ch, args.vocab_size, args.latent_size, args.dropout, args.embedding_size]
    config = ""
    for i in map(str, config_list):
        config = config + '_' + i
    args.config = config
    return args


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def main(args):
    writer = SummaryWriter(args.log_dir + args.timestamp + args.config)
    device = torch.device('cuda:0' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    # mp.set_start_method('spawn')
    # Model
    model = AttnGAN(embedding_size=args.embedding_size, latent_size=args.latent_size, in_ch=args.in_ch, vocab_size=args.vocab_size, hidden_size=args.hidden_size, num_layer=1, dropout=args.dropout)
    # test_dict = [name for name, module in model.named_children() if len(list(module.parameters())) == 0]
    # print(test_dict)
    # print(test_dict)
    # A logger to store training process information
    train_logger = Logger()

    # Specifying loss function, metric(s), and optimizer
    loss = {
        'gan' : gan_loss,
        'kld' : kld_loss,
        'damsm' : DAMSM_loss
    }
    metrics = []
    optimizer = {
        # name : optim.Adam(module.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=True, betas=(0.5, 0.999))
        name : optim.RMSprop(nn.ParameterList(module.parameters()), lr=args.lr, alpha=0.99, eps=1e-08, weight_decay=args.wd)
        for name, module in model.named_children()
    }
    print(count_parameters(model))
    # Data loader and validation split
    data_loader = CubDataLoader('../data/birds', args.batch_size, args.valid_batch_size, args.validation_split, args.validation_fold, shuffle=True, num_workers=0)
    # data_loader = CocoDataLoader('../cocoapi', args.batch_size, args.valid_batch_size, args.validation_split, args.validation_fold, shuffle=True, num_workers=4)
    valid_data_loader = data_loader.get_valid_loader()

    # An identifier for this training session
    training_name = type(model).__name__

    # Trainer instance
    trainer = Trainer(model, loss, metrics,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      optimizer=optimizer,
                      epochs=args.epochs,
                      train_logger=train_logger,
                      writer=writer,
                      save_dir=args.log_dir + args.timestamp + args.config,
                      save_freq=args.save_freq,
                      resume=args.resume,
                      verbosity=args.verbosity,
                      training_name=training_name,
                      device=device,
                      monitor='loss',
                      monitor_mode='min')

    # Start training!
    trainer.train()

    # See training history
    print(train_logger)


if __name__ == '__main__':
    main(arg_parse())

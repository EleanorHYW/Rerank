"""

Pytorch implementation of Pointer Network.

http://arxiv.org/pdf/1506.03134v1.pdf.

"""

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import logging
import os
import shutil
import sys
import numpy as np
import argparse

from PointerNet import PointerNet
from Seq2SlateLoss import Seq2SlateLoss
from RerankingDataset import RerankingDataset
from utils import BatchManager, load_data, load_embedding
from tensorboardX import SummaryWriter
from Metrics import auc, getmapk
from sklearn.metrics import roc_auc_score


parser = argparse.ArgumentParser(description="Pytorch implementation of Pointer-Net")

# Data
parser.add_argument('--train_size', default=1000000, type=int, help='Training data size')
parser.add_argument('--valid_size', default=10000, type=int, help='Validation data size')
parser.add_argument('--eval_size', default=10000, type=int, help='Evaluation data size')
parser.add_argument('--batch_size', default=256, type=int, help='Batch size')
parser.add_argument('--max_len', default=50, type=int, help='max item sequence length')
parser.add_argument('--history_len', default=5, type=int, help='history sequence length')
# Train
parser.add_argument('--n_epochs', default=50000, type=int, help='Number of epochs')
# parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
# GPU
parser.add_argument('--gpu', default=True, action='store_true', help='Enable gpu')
# Network
parser.add_argument('--embedding_size', type=int, default=128, help='Embedding size')
parser.add_argument('--hiddens', type=int, default=512, help='Number of hidden units')
parser.add_argument('--n_lstms', type=int, default=2, help='Number of LSTM layers')
parser.add_argument('--dropout', type=float, default=0., help='Dropout value')
parser.add_argument('--bidir', default=False, action='store_true', help='Bidirectional')
parser.add_argument('--seed', type=int, default=1, help='seed')
parser.add_argument('--ckpt_file', type=str, default='checkpoint_0.pkl')

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    filename='log/train.log',
    filemode='w',
)

def eval(args, model, eval_dataloader):
    model.eval()
    average_auc = 0
    average_mapk = 0
    limit = len(eval_dataloader)
    average_over_num = limit
    logging.info('Evaluating on %d minibatches...' % limit)
    with torch.no_grad():
        for idx, sample in enumerate(eval_dataloader):
            logits, indices = model(sample['Feeds'], sample['History'])
            labels = sample['Labels']
            labels = labels.view(args.batch_size, -1)
            true_labels = torch.gather(labels, 1, indices)
            scores = torch.max(logits, 2)[0]
            # if a class is not present in the ground truth of a batch, ROC AUC score is not defined in that case.
            # may raise 'ValueError: Only one class present in y_true' that way
            try:
                auc_score = roc_auc_score(true_labels, scores)
                average_auc += auc_score
                # logging.info('eval step %d, auc_score = %f' % (idx + 1, auc_score))
            except ValueError:
                average_over_num -= 1
                pass
            # average_auc += auc(true_labels, indices)
            # actual = torch.tensor(range(0, 10)).repeat(args.batch_size, 1)
            # actual[labels == 0] = 11
            # average_mapk += getmapk(actual, indices)
            if idx >= limit:
                break
        average_auc /= average_over_num
        print("average_auc: ", average_auc)
    model.train()

def validation(args, model, valid_dataloader, SSLloss, train_loss, idxa, epoch, writer):
    model.eval()
    valid_loss = 0
    cnt = len(valid_dataloader)
    for idx, sample in enumerate(valid_dataloader):
        with torch.no_grad():
            logits, indices = model(sample['Feeds'], sample['History'])
            loss = SSLloss(logits, sample['Labels'])
            if torch.isnan(loss):
                cnt -= 1
            else:
                valid_loss += SSLloss(logits, sample['Labels'])
    logging.info('During validation, %d in %d batches loss is nan' % (len(valid_dataloader) - cnt, len(valid_dataloader)))
    if cnt != 0:
        valid_loss /= cnt
        logging.info('epoch %d, after step %d, begin validation, average validation loss = %f'
                 % (epoch, idxa + 1, valid_loss))
    else:
        valid_loss = float('inf')
    writer.add_scalar('scalar/train_loss', train_loss, (idxa + 1) // 50)
    writer.add_scalar('scalar/valid_loss', valid_loss, (idxa + 1) // 50)
    model.train()
    torch.cuda.empty_cache()

def train(args, model, train_dataloader, valid_dataloader, eval_dataloader, optimizer, scheduler, start_epoch=0):
    logging.info("Start to train with lr=%f..." % optimizer.param_groups[0]['lr'])

    model.train()
    for epoch in range(start_epoch, args.n_epochs):
        if os.path.isdir('runs/epoch%d' % epoch):
            shutil.rmtree('runs/epoch%d' % epoch)
        writer = SummaryWriter('runs/epoch%d' % epoch)

        for idx, sample in enumerate(train_dataloader):
            SSLloss = Seq2SlateLoss()
            logits, indices = model(sample['Feeds'], sample['History'])
            logits = logits.contiguous()

            optimizer.zero_grad()
            loss = SSLloss(logits, sample['Labels'])
            # print(loss)
            loss.backward()  # do not use retain_graph=True
            # torch.nn.utils.clip_grad_value_(model.parameters(), 5)
            optimizer.step()
            if (idx + 1) % 10 == 0:
                logging.info('epoch %d, step %d, loss = %f' % (epoch, idx + 1, loss))
            if (idx + 1) % 50 == 0:
                train_loss = loss.cpu().detach().numpy()
                validation(args, model, valid_dataloader, SSLloss, train_loss, idx, epoch, writer)
            if (idx + 1) % 100 == 0:
                eval(args, model, eval_dataloader)
            scheduler.step()  # last_epoch=-1, which will not update lr at the first time

        if epoch < 4:
            scheduler.step()  # make sure lr will not be too small
        save_state = {'state_dict': model.state_dict(),
                      'epoch': epoch + 1,
                      'lr': optimizer.param_groups[0]['lr']}
        if not os.path.exists('./models'):
            os.mkdir('./models')
        torch.save(save_state, './models/checkpoint_%d.pkl' % epoch)
        logging.info('Model saved in dir %s' % './models')
        writer.close()

def main():
    # define a new Handler to log to console as well
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    Args = parser.parse_args()
    print(Args)

    if Args.gpu and torch.cuda.is_available():
        USE_CUDA = True
        print('Using GPU, %i devices.' % torch.cuda.device_count())
    else:
        USE_CUDA = False

    model = PointerNet(Args.embedding_size,
                       Args.hiddens,
                       Args.n_lstms,
                       Args.dropout,
                       Args.bidir)
    print(model)

    if USE_CUDA:
        model.cuda()
        net = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    max_len = Args.max_len
    bsz = Args.batch_size
    history_len = Args.history_len
    # import pdb; pdb.set_trace()
    data_dict = {'train': RerankingDataset(Args.train_size, max_len, history_len),
               'valid': RerankingDataset(Args.valid_size, max_len, history_len),
               'eval': RerankingDataset(Args.eval_size, max_len, history_len),
    }

    train_data = data_dict['train']
    train_dataloader = DataLoader(train_data,
                                  batch_size=bsz,
                                  shuffle=True,
                                  num_workers=4,
                                  drop_last=True)
    valid_data = data_dict['valid']
    valid_dataloader = DataLoader(valid_data,
                                  batch_size=bsz,
                                  shuffle=True,
                                  num_workers=4,
                                  drop_last=True)
    eval_data = data_dict['eval']
    eval_dataloader = DataLoader(eval_data,
                                  batch_size=bsz,
                                  shuffle=True,
                                  num_workers=4,
                                  drop_last=True)

    saved_state = {'epoch': 0, 'lr': 0.001}
    if os.path.exists(Args.ckpt_file):
        saved_state = torch.load(Args.ckpt_file)
        model.load_state_dict(saved_state['state_dict'])
        logging.info('Load model parameters from %s' % Args.ckpt_file)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=saved_state['lr'])
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=saved_state['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    # scheduler.step()  # last_epoch=-1, which will not update lr at the first time

    train(Args, model, train_dataloader, valid_dataloader, eval_dataloader, optimizer, scheduler, saved_state['epoch'])
    # eval(Args, model, eval_dataloader)

if __name__ == '__main__':
    main()

"""

Pytorch implementation of Pointer Network.

http://arxiv.org/pdf/1506.03134v1.pdf.

"""

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
import logging
import os
import shutil
import sys
import numpy as np
import argparse
from torch import randperm
from torch._utils import _accumulate
import json

from PointerNet import PointerNet
from PNet import PNet
from Seq2SlateLoss import Seq2SlateLoss
from RerankingDataset import MslrItemDataset, MslrLabelDataset, RightPadDataset, LengthDataset, NestedDictionaryDataset, SubsetDataset, MaskDataset
from utils import load_data, random_split, delete_seq_with_max_length, feature_normalize
from tensorboardX import SummaryWriter
from Metrics import auc, getmapk, getndcgK
from sklearn.metrics import roc_auc_score
from test_onnx import test


parser = argparse.ArgumentParser(description="Pytorch implementation of Pointer-Net")

# Data
parser.add_argument('--train_size', default=0.8, type=float, help='Training data size')
parser.add_argument('--valid_size', default=0.1, type=float, help='Validation data size')
parser.add_argument('--eval_size', default=0.1, type=float, help='Evaluation data size')
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

def eval(args, model, eval_dataloader, limit=None):
    model.eval()
    ndcg = 0
    ndcg_ten = 0
    cnt = 0
    limit = len(eval_dataloader) if limit is None else limit
    logging.info('Evaluating on %d minibatches...' % limit)
    with torch.no_grad():
        for idx, sample in enumerate(eval_dataloader):
            logits, indices, atts = model(sample['Feeds'], sample['Masks'])
            # evaluation needs relevance_score related label
            scores = sample['Scores']
            golden_list = torch.sort(scores, dim=-1, descending=True)[0]
            for i in range(len(indices)):
                model_list = torch.gather(scores[i].cuda(), 0, indices[i].squeeze(0))
                length = torch.nonzero(golden_list[0])[-1] + 1
                ndcg += getndcgK(model_list, golden_list[i], model_list.size(0))
                ndcg_ten += getndcgK(model_list, golden_list[i], 10)
                cnt += 1
            if idx >= limit:
                break
        ndcg = ndcg / cnt
        ndcg_ten = ndcg_ten / cnt
        print("ndcg: ", ndcg.item())
        print("ndcg10:", ndcg_ten.item())
    model.train()

def validation(args, model, valid_dataloader, SSLloss, train_loss, idxa, epoch, writer):
    model.eval()
    valid_loss = 0
    cnt = len(valid_dataloader)
    for idx, sample in enumerate(valid_dataloader):
        with torch.no_grad():
            logits, indices, atts = model(sample['Feeds'], sample['Masks'])
            loss = SSLloss(logits, sample['Labels'])
            valid_loss += loss
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
            # import pdb; pdb.set_trace()
            optimizer.zero_grad()
            SSLloss = Seq2SlateLoss(args.max_len)
            logits, indices, atts = model(sample['Feeds'], sample['Masks'])
            loss = SSLloss(logits, sample['Labels'])
            loss.backward()  # do not use retain_graph=True
            optimizer.step()
            # if idx >= 0:
            #     for item in model.named_parameters():
            #         print(item[0])
            #         print(item[1].grad)
            # torch.nn.utils.clip_grad_value_(model.parameters(), 5)
            if (idx + 1) % 5 == 0:
                logging.info('epoch %d, step %d, loss = %f' % (epoch, idx + 1, loss))
            if (idx + 1) % 300 == 0:
                train_loss = loss.cpu().detach().numpy()
                validation(args, model, valid_dataloader, SSLloss, train_loss, idx, epoch, writer)
            if (idx + 1) % 20 == 0:
                # for name, param in model.named_parameters():
                #     print(name)
                #     print(param)
                #     import pdb; pdb.set_trace()
                eval(args, model, eval_dataloader)

        if epoch < 10 and epoch % 3 == 0:
            scheduler.step()  # make sure lr will not be too small
        save_state = {'state_dict': model.state_dict(),
                      'epoch': epoch + 1,
                      'lr': optimizer.param_groups[0]['lr']}
        if not os.path.exists('./models'):
            os.mkdir('./models')
        torch.save(save_state, './models/checkpoint_%d.pt' % epoch)
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

    data = np.load('../dict_large.npz', allow_pickle=True)
    feeds, labels, scores = delete_seq_with_max_length(data['Feeds'], data['Labels'], data['Scores'], max_len)
    feeds = feature_normalize(feeds)
    dataset = NestedDictionaryDataset({
        'Feeds': MslrItemDataset(feeds),
        'Labels': MslrLabelDataset(labels),
        'Masks': MaskDataset(labels),
        'Scores': MslrLabelDataset(scores),
    })

    data_size = len(feeds)
    train_size = int(Args.train_size * data_size)
    valid_size = int(Args.valid_size * data_size)
    test_size = int(data_size - train_size - valid_size)
    lengths = [train_size, valid_size, test_size]
    assert sum(lengths) == data_size, "Sum of input lengths does not equal the length of the input dataset!"
    indices = randperm(sum(lengths)).tolist()
    train_dataset, valid_dataset, test_dataset = [SubsetDataset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]
    logging.info("train dataset: {} samples, valid dataset: {} samples, test dataset: {} samples".format(len(train_dataset), len(valid_dataset), len(test_dataset)))

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=bsz,
                                  shuffle=False,
                                  num_workers=0,
                                  collate_fn=train_dataset.collater,
                                  drop_last=True)
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=bsz,
                                  shuffle=False,
                                  num_workers=0,
                                  collate_fn=valid_dataset.collater,
                                  drop_last=True)
    eval_dataloader = DataLoader(test_dataset,
                                  batch_size=bsz,
                                  shuffle=False,
                                  num_workers=0,
                                  collate_fn=test_dataset.collater,
                                  drop_last=True)

    saved_state = {'epoch': 0, 'lr': 0.001}
    if os.path.exists(Args.ckpt_file):
        saved_state = torch.load(Args.ckpt_file)
        model.load_state_dict(saved_state['state_dict'])
        logging.info('Load model parameters from %s' % Args.ckpt_file)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=saved_state['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    # import pdb; pdb.set_trace()
    train(Args, model, train_dataloader, valid_dataloader, eval_dataloader, optimizer, scheduler, saved_state['epoch'])
    eval(Args, model, eval_dataloader)
    # test(model, './models/checkpoint_0.pt', 'encoder.onnx', 'decoder.onnx')

if __name__ == '__main__':
    main()

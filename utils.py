import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import numpy as np
from PointerNet import PointerNet
from Data_Generator import TSPDataset
from Seq2SlateLoss import Seq2SlateLoss
from RerankingDataset import RerankingDataset

class BatchManager:
    def __init__(self, data, batch_size):
        self.steps = int(len(data) / batch_size)
        # comment following two lines to neglect the last batch
        if self.steps * batch_size < len(data):
            self.steps += 1
        self.data = data
        self.batch_size = batch_size
        self.bid = 0

    def next_batch(self):
        stncs = list(self.data[self.bid * self.batch_size: (self.bid + 1) * self.batch_size])
        self.bid += 1
        if self.bid == self.steps:
            self.bid = 0
        return stncs

def load_data(filename, max_len, n_lines=None):
    """
    :param filename: the file to read
    :param max_len: maximum length of a sequence
    :return: datas
    """
    fin = open(filename, "r", encoding="utf8")
    datas = []
    for idx, line in enumerate(fin):
        if line == '' or idx == n_lines:
            break
        words = line.strip().split()
        if len(words) > max_len - 2:
            words = words[:max_len-2]
        datas.append(words)
    return datas

def load_embedding(item2emb, item_lists):
    embeddings = []
    for item_list in item_lists:
        embedding = []
        for item in item_list:
            embedding.append(item2emb[item])
        embeddings.append(embedding)
    return embeddings

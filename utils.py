import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import randperm
from torch._utils import _accumulate
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pack_sequence,pad_packed_sequence

import math
import numpy as np
from PointerNet import PointerNet
from Seq2SlateLoss import Seq2SlateLoss

def load_data(filename):
        print("loading data from", filename)
        f = open(filename, 'r')
        lines = f.readlines()
        instances = []
        labels = []
        curInstance = []
        curLabel = []
        prevInstanceNum = 0
        for line in lines:
            item = eval(line.strip())
            if item['instanceNum'] != prevInstanceNum:
                instances.append(np.array(curInstance))
                labels.append(np.array(curLabel))
                curInstance = [np.array(item['embedding'])]
                curLabel = [item['label']]
                prevInstanceNum = item['instanceNum']
            else:
                curInstance.append(np.array(item['embedding']))
                curLabel.append(item['label'])
        instances.append(np.array(curInstance))
        labels.append(np.array(curLabel))

        print("Load data finished!")
        f.close()

        return {
            'Feeds': np.array(instances[1:]),
            'Labels': np.array(labels[1:]),
        }

def delete_seq_with_max_length(data, labels, scores, max_length):
    ret_data = []
    ret_labels = []
    ret_scores = []
    cnt = 0
    for i in range(len(data)):
        if len(data[i]) > max_length:
            cnt += 1
        else:
            ret_data.append(data[i])
            ret_labels.append(labels[i])
            ret_scores.append(scores[i])

    print("Delete {} sequence with length > {}".format(cnt, max_length))
    return np.array(ret_data), np.array(ret_labels), np.array(ret_scores)

def feature_normalize(feeds):
    for i in range(len(feeds)):
        mu = np.mean(feeds[i], axis=0).reshape(-1)
        std = np.std(feeds[i], axis=0).reshape(-1)
        feeds[i] = (feeds[i] - mu)/(std + 1/math.exp(1))
    return feeds

def copy_tensor(src, dst):
    assert dst.numel() == src.numel()
    dst.copy_(src)

def collate_tokens(data, pad_to_length=None, pad_idx=0):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    # values: bsz * seq_len * feature_dim
    # size: max_seq_len
    # size = max(v.size(0) for v in values)
    # size = size if pad_to_length is None else max(size, pad_to_length)
    # # res: bsz * size * feature_dim
    # res = values[0].new(len(values), size, values[0].size(1)).fill_(pad_idx)
    data.sort(key=lambda x: len(x), reverse=True)
    data = pack_sequence(data)
    return data

def collate_labels(data, pad_to_length=None, pad_idx=0):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    # values: bsz * seq_len
    # size: max_seq_len
    # size = max(v.size(0) for v in values)
    # size = size if pad_to_length is None else max(size, pad_to_length)
    # # res: bsz * size
    # res = values[0].new(len(values), size).fill_(pad_idx)

    # for i, v in enumerate(values):
    #     copy_tensor(v, res[i][:len(v)])
    # return res
    data.sort(key=lambda x: len(x), reverse=True)
    # data = pack_sequence(data)
    # seq_len = [s.size(0) for s in data]
    data = pad_sequence(data, batch_first=True)    
    # data = pack_padded_sequence(data, seq_len, batch_first=True)
    return data

def collate_masks(values, pad_to_length=None, pad_idx=0):
    values.sort(key=lambda x: len(x), reverse=True)
    size = max(len(v) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    res = values[0].new(len(values), size).fill_(pad_idx)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][:len(v)])
    return res

def random_split(dataset, lengths):
    assert sum(lengths) == len(dataset), "Sum of input lengths does not equal the length of the input dataset!"
    indices = randperm(sum(lengths)).tolist()
    split_list = [indices[offset - length : offset] for offset, length in zip(_accumulate(lengths), lengths)]
    return split_list

def load_embedding(item2emb, item_lists):
    embeddings = []
    for item_list in item_lists:
        embedding = []
        for item in item_list:
            embedding.append(item2emb[item])
        embeddings.append(embedding)
    return embeddings

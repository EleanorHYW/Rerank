import torch
from torch.utils.data import Dataset
from utils import collate_tokens, collate_labels, collate_masks
import numpy as np
import itertools
from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data.dataloader import default_collate


def _flatten(dico, prefix=None):
    """Flatten a nested dictionary."""
    new_dico = OrderedDict()
    if isinstance(dico, dict):
        prefix = prefix + '.' if prefix is not None else ''
        for k, v in dico.items():
            if v is None:
                continue
            new_dico.update(_flatten(v, prefix + k))
    elif isinstance(dico, list):
        for i, v in enumerate(dico):
            new_dico.update(_flatten(v, prefix + '.[' + str(i) + ']'))
    else:
        new_dico = OrderedDict({prefix: dico})
    return new_dico


def _unflatten(dico):
    """Unflatten a flattened dictionary into a nested dictionary."""
    new_dico = OrderedDict()
    for full_k, v in dico.items():
        full_k = full_k.split('.')
        node = new_dico
        for k in full_k[:-1]:
            if k.startswith('[') and k.endswith(']'):
                k = int(k[1:-1])
            if k not in node:
                node[k] = OrderedDict()
            node = node[k]
        node[full_k[-1]] = v
    return new_dico

class RerankingDataset(Dataset):

    def __init__(self, data_size, seq_len, history_len):
        self.data_size = data_size
        self.seq_len = seq_len
        self.history_len = history_len
        self.data = self._generate_data()

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        feeds = torch.from_numpy(self.data['Feeds'][idx]).float()
        labels = torch.from_numpy(self.data['Labels'][idx]).long()
        history = torch.from_numpy(self.data['History'][idx]).float()
        sample = {
            'Feeds': feeds,
            'Labels': labels,
            'History': history,
            }
        return sample

    def _generate_data(self):
        # import pdb; pdb.set_trace()
        rerank_lists = []
        labels = []
        history = []
        for i in range(self.data_size):
            rerank_lists.append(np.random.random((self.seq_len, 4)))
            labels.append(np.random.randint(2, size=(self.seq_len, 1)))
            history.append(np.random.random((self.history_len, 4)))
        print("Generate data finished!")
        return {
            'Feeds': rerank_lists,
            'Labels': labels,
            'History': history,
            }

    def _to1hotvec(self, items):
        """
        :param points: List of integers representing the points indexes
        :return: Matrix of One-Hot vectors
        """
        vec = np.zeros((len(items), self.seq_len))
        for i, v in enumerate(vec):
            v[items[i]] = 1

        return vec

class MslrItemDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).float()

    def collater(self, samples):
        return collate_tokens(samples)

class MslrLabelDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).long()

    def collater(self, samples):
        return collate_labels(samples)

class RightPadDataset(Dataset):

    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collater(self, samples):
        return collate_tokens(samples)

class NumelDataset(Dataset):

    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        item = self.data[index]
        if torch.is_tensor(item):
            return torch.numel(item)
        else:
            return np.size(item)

    def __len__(self):
        return len(self.data)

    def collater(self, samples):
        return torch.tensor(samples)

class LengthDataset(Dataset):

    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        item = self.data[index]
        if torch.is_tensor(item):
            return item.size(0)
        else:
            return len(item)

    def __len__(self):
        return len(self.data)

    def collater(self, samples):
        return torch.tensor(samples)

class NestedDictionaryDataset(Dataset):

    def __init__(self, defn):
        super().__init__()
        self.defn = _flatten(defn)

        first = None
        for v in self.defn.values():
            first = first or v
            if len(v) > 0:
                assert len(v) == len(first), 'dataset lengths must match'

        self._len = len(first)
    
    def __getitem__(self, index):
        return OrderedDict((k, ds[index]) for k, ds in self.defn.items())

    def __len__(self):
        return self._len

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch suitable for forwarding with a Model
        """
        if len(samples) == 0:
            return {}
        sample = OrderedDict()
        for k, ds in self.defn.items():
            try:
                sample[k] = ds.collater([s[k] for s in samples])
            except NotImplementedError:
                sample[k] = default_collate([s[k] for s in samples])
        return _unflatten(sample)

class SubsetDataset(Dataset):
    def __init__(self, data, indices):
        self.data = data
        self.indices = indices

    def __getitem__(self, idx):
        return self.data[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def collater(self, samples):
        return self.data.collater(samples)

class MaskDataset(Dataset):
    def __init__(self, data):
        self.data = data.copy()
        self.len = len(data)
        for i in range(self.len):
            self.data[i] = np.ones_like(self.data[i])

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])

    def __len__(self):
        return self.len

    def collater(self, samples):
        return collate_masks(samples)
            

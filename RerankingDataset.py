import torch
from torch.utils.data import Dataset
import numpy as np
import itertools
from tqdm import tqdm

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

import torch
from torch.utils.data import Dataset

import numpy as np
import gzip
import os

def get_file_name(type, filenumber):
    return 'data/1/replay_logs/$store$_{}_ckpt.{}.gz'.format(type, filenumber)


class AtariDataset(Dataset):

    def __init__(self, base_path, file_id, seq_length):
        g = gzip.GzipFile(filename=os.path.join(base_path, 'obs-{}.npy.gz'.format(file_id)))
        self.obs = np.load(g)

        self.reward = np.load(os.path.join(base_path, 'reward-{}.npy'.format(file_id)))
        self.action = np.load(os.path.join(base_path, 'action-{}.npy'.format(file_id)))
        terminal = np.load(os.path.join(base_path, 'terminal-{}.npy'.format(file_id)))

        self.ret = np.load(os.path.join(base_path, 'return-{}.npy'.format(file_id)))
        self.done_idx = np.where(terminal == 1)[0]
        self.seq_length = seq_length
        self.idx_list = []

        j = 0
        curr_end = self.done_idx[j]-seq_length
        i = 0
        while i < len(terminal)-seq_length:
            if i < curr_end:
                self.idx_list.append(i)
            else:
                j += 1
                i = curr_end+seq_length
                if j < len(self.done_idx):
                    curr_end = self.done_idx[j]-seq_length
                else:
                    curr_end = len(terminal)-seq_length

            i += 1


    def __getitem__(self, item):
        idx = self.idx_list[item]
        end = idx + self.seq_length

        return self.obs[idx:end], self.ret[idx: end], self.action[idx:end], self.reward[idx:end]

    def __len__(self):
        return len(self.idx_list)
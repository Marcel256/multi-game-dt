import numpy as np

from atari_dataset import AtariDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import gzip
import os

def get_file_name(root, type, filenumber):
    file = '$store$_{}_ckpt.{}.gz'.format(type, filenumber)
    return os.path.join(root, file)



g = gzip.GzipFile(filename=get_file_name('data/download/Asterix', 'observation', 50))
obs = np.load(g)


plt.imshow(obs[5])
plt.show()
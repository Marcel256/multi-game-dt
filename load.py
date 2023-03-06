import gzip

import numpy as np
import matplotlib.pyplot as plt
import pickle

def get_file_name(type, filenumber):
    return 'data/1/replay_logs/$store$_{}_ckpt.{}.gz'.format(type, filenumber)

num_buffers = 50



g = gzip.GzipFile(filename=get_file_name('action', 1))
obs = np.load(g)



print(obs[-100:])
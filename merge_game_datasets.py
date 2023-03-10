import gzip
import os

import numpy as np
splits = 1
games = [
'Pong',
'Qbert',
'DemonAttack',
'SpaceInvaders',
'Breakout',
]

input_dir = 'data/split'
out_dir = 'data/merged'

files = ['obs-{}.npy.gz', 'action-{}.npy', 'reward-{}.npy', 'return-{}.npy', 'terminal-{}.npy']


def load_file(file_name, gz):
    if gz:
        with gzip.GzipFile(file_name, 'r') as fh:
            content = np.load(fh)
    else:
        with open(file_name, 'rb') as fh:
            content = np.load(fh)

    return content


def save_to_file(arr, file_name, gz):
    if gz:
        with gzip.GzipFile(file_name, 'w') as fh:
            np.save(file=fh, arr=arr)
    else:
        with open(file_name, 'wb') as fh:
            np.save(file=fh, arr=arr)


for split in range(splits):
    for file in files:
        data = []
        file_name = file.format(split)
        print('Merging ', file_name)
        gz = file_name.endswith('gz')
        for game in games:
            path = os.path.join(input_dir, game, file_name)
            data.append(load_file(path, gz))

        out_path = os.path.join(out_dir, file_name)
        save_to_file(np.concatenate(data, axis=0), out_path, gz)
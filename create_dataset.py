import numpy as np
import gzip
import os

def get_file_name(root_dir, type, file_number):
    return '{}/$store$_{}_ckpt.{}.gz'.format(root_dir, type, file_number)


def split_buffers(root_dir, target_dir):

    num_buffers = 50

    split_id = 0
    for buffer in range(1, 2):
        g = gzip.GzipFile(filename=get_file_name(root_dir, 'observation', buffer))
        data = np.load(g)
        splits = np.split(data, 10)

        for split in splits:
            with open(os.path.join(target_dir, 'observation-{}.npy'.format(split_id)), 'wb') as fh:
                np.save(fh, split)
            split_id += 1





def create_return_files(root_dir, target_dir):
    num_buffers = 50

    curr_ret = 0

    for buffer in range(num_buffers, 0, -1):
        g = gzip.GzipFile(filename=get_file_name(root_dir, 'terminal', buffer))
        terminal = np.load(g)

        g = gzip.GzipFile(filename=get_file_name(root_dir, 'reward', buffer))
        reward = np.load(g)
        ret = np.zeros((len(reward)), dtype=np.int16)

        for i in reversed(range(0, len(ret))):
            if terminal[i] == 1:
                curr_ret = 0
            curr_ret += reward[i]
            ret[i] = curr_ret

        with open(os.path.join(target_dir, 'return-{}.npy'.format(buffer)), 'wb') as fh:
            np.save(fh, ret)






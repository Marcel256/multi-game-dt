import os
import gzip

import numpy as np
import random


def get_file_name(root, type, filenumber):
    file = '$store$_{}_ckpt.{}.gz'.format(type, filenumber)
    return os.path.join(root, file)


def create_sampled_array(data, out, trajectories):
    out_start = 0
    for traj in trajectories:
        start, end = traj
        length = end - start
        out[out_start: (out_start + length)] = data[start:end]
        out_start += length


def save_array(data, file):
    with open(file, 'wb') as fh:
        np.save(fh, data)

root_dir = 'data/breakout/1/replay_logs'
out_dir = ''

sample_prob = 0.02

num_buffers = 1

curr_ret = 0



for buffer in range(num_buffers, 0, -1):
    g = gzip.GzipFile(filename=get_file_name(root_dir, 'terminal', buffer))
    terminal = np.load(g)
    end_idx = np.where(terminal == 1)[0]
    start = 0
    sampled_traj = []
    total_transitions = 0
    for i in range(len(end_idx)):
        if random.random() < sample_prob:
            sampled_traj.append((start, end_idx[i]+1))
            total_transitions += end_idx[i]+1 - start

        start = end_idx[i]+1

    random.shuffle(sampled_traj)
    out = np.empty((total_transitions,), dtype=np.uint8)
    create_sampled_array(terminal, out, sampled_traj)
    filename = os.path.join(out_dir, 'terminal-{}.npy'.format(buffer))
    save_array(out, filename)

    out = np.empty((total_transitions, 84, 84), dtype=np.uint8)
    create_sampled_array(terminal, out, sampled_traj)
    filename = os.path.join(out_dir, 'obs-{}.npy'.format(buffer))
    save_array(out, filename)

    out = np.empty((total_transitions,), dtype=np.uint8)
    create_sampled_array(terminal, out, sampled_traj)
    filename = os.path.join(out_dir, 'action-{}.npy'.format(buffer))
    save_array(out, filename)

    out = np.empty((total_transitions,), dtype=np.uint8)
    create_sampled_array(terminal, out, sampled_traj)
    filename = os.path.join(out_dir, 'reward-{}.npy'.format(buffer))
    save_array(out, filename)

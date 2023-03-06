import os
import gzip

import numpy as np
from atari_action_utils import LIMITED_ACTION_TO_FULL_ACTION


def get_file_name(root, type, filenumber):
    file = '$store$_{}_ckpt.{}.gz'.format(type, filenumber)
    return os.path.join(root, file)


root_dir = 'data/download'
out_base_dir = 'data/single_split'

def split_buffer(file_number, base_dir, out_dir, splits, game):

    g = gzip.GzipFile(filename=get_file_name(base_dir, 'terminal', file_number))
    terminal = np.load(g)
    end_idx = np.where(terminal == 1)[0]

    g = gzip.GzipFile(filename=get_file_name(base_dir, 'observation', file_number))
    obs = np.load(g)

    g = gzip.GzipFile(filename=get_file_name(base_dir, 'action', file_number))
    action = np.load(g)

    g = gzip.GzipFile(filename=get_file_name(base_dir, 'reward', file_number))
    reward = np.load(g)

    ret = np.zeros_like(reward)

    curr_ret = 0

    reward = np.clip(reward, -1, 1)

    for i in range(end_idx[-1], -1, -1):
        if terminal[i] == 1:
            curr_ret = 0
        curr_ret += reward[i]
        ret[i] = curr_ret
        action[i] = LIMITED_ACTION_TO_FULL_ACTION[game][action[i]]
    split_length = end_idx[-1] // splits
    start = 0
    i = 0
    while start < end_idx[-1]:
        end = end_idx[-1]
        for idx in end_idx:
            length = idx+1-start
            if length > split_length:
                end = idx+1
                break
        end = min(end_idx[-1]+1, end)
        data = obs[start : end]
        print(start,' ',end)
        with gzip.GzipFile(os.path.join(out_dir, 'obs-{}.npy.gz'.format(i)), "w") as fh:
            np.save(file=fh, arr=data)

        with open(os.path.join(out_dir, 'action-{}.npy'.format(i)), 'wb') as fh:
            np.save(file=fh, arr=action[start:end])

        with open(os.path.join(out_dir, 'reward-{}.npy'.format(i)), 'wb') as fh:
            np.save(file=fh, arr=reward[start:end])

        with open(os.path.join(out_dir, 'return-{}.npy'.format(i)), 'wb') as fh:
            np.save(file=fh, arr=ret[start:end])

        with open(os.path.join(out_dir, 'terminal-{}.npy'.format(i)), 'wb') as fh:
            np.save(file=fh, arr=terminal[start:end])
        i += 1
        start = end


games = ['Asterix',
'BeamRider',
'Breakout',
'DemonAttack',
'Gravitar',
'TimePilot',
'SpaceInvaders',
'Jamesbond',
'Assault',
'Frostbite']

ogames = ['Breakout']

for game in ogames:
    base = os.path.join(root_dir, game)
    out = os.path.join(out_base_dir, game)
    if not os.path.exists(out):
        os.makedirs(out)
    split_buffer(50, base, out, 1, game)
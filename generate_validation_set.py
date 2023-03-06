import os
import gzip

import numpy as np

from atari_action_utils import LIMITED_ACTION_TO_FULL_ACTION

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

ogames = ['Assault']

def get_file_name(root_dir, type, file_number):
    return os.path.join(root_dir, '$store$_{}_ckpt.{}.gz'.format(type, file_number))


root_dir = 'data/download/'
file_number = 41

result_obs = []
result_action = []
result_reward = []
result_return = []
result_terminal = []

for game in ogames:
    print(game)
    base_dir = os.path.join(root_dir, game)
    g = gzip.GzipFile(filename=get_file_name(base_dir, 'terminal', file_number))
    terminal = np.load(g)
    end_idx = np.where(terminal == 1)[0]

    g = gzip.GzipFile(filename=get_file_name(base_dir, 'observation', file_number))
    obs = np.load(g)

    g = gzip.GzipFile(filename=get_file_name(base_dir, 'action', file_number))
    action = np.load(g)

    g = gzip.GzipFile(filename=get_file_name(base_dir, 'reward', file_number))
    reward = np.load(g)

    # Use first trajectory
    start = end_idx[3]
    end = end_idx[4]+1
    result_obs.append(np.copy(obs[start: end]))
    res_action = np.copy(action[start: end])
    r = np.clip(np.copy(reward[start: end]), -1, 1)
    result_reward.append(r)
    result_terminal.append(np.copy(terminal[start: end]))

    ret = np.zeros_like(r)
    curr_ret = 0
    for i in range(len(r)-1, -1, -1):
        if terminal[i] == 1:
            curr_ret = 0
        curr_ret += r[i]
        ret[i] = curr_ret
        res_action[i] = LIMITED_ACTION_TO_FULL_ACTION[game][res_action[i]]

    result_return.append(ret)
    result_action.append(res_action)
    del obs
    del action
    del reward
    del ret
    del terminal



out_dir = 'data/valid_assault'

with gzip.GzipFile(os.path.join(out_dir, 'obs-0.npy.gz'), 'w') as fh:
    np.save(file=fh, arr=np.concatenate(result_obs, axis=0))

with open(os.path.join(out_dir, 'action-0.npy'), 'wb') as fh:
    np.save(file=fh, arr=np.concatenate(result_action, axis=0))

with open(os.path.join(out_dir, 'reward-0.npy'), 'wb') as fh:
    np.save(file=fh, arr=np.concatenate(result_reward, axis=0))

with open(os.path.join(out_dir, 'return-0.npy'), 'wb') as fh:
    np.save(file=fh, arr=np.concatenate(result_return, axis=0))

with open(os.path.join(out_dir, 'terminal-0.npy'), 'wb') as fh:
    np.save(file=fh, arr=np.concatenate(result_terminal, axis=0))
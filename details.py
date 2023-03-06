import os

import numpy as np

base_path = 'data/single_split/Breakout'


for file_id in range(1):
    ret = np.load(os.path.join(base_path, 'return-{}.npy'.format(file_id)))
    terminal = np.load(os.path.join(base_path, 'terminal-{}.npy'.format(file_id)))
    reward = np.load(os.path.join(base_path, 'reward-{}.npy'.format(file_id)))
    action = np.load(os.path.join(base_path, 'action-{}.npy'.format(file_id)))
    done_idx = np.where(terminal == 1)[0]
    min_ret = 9999
    max_ret = -9999
    returns = []
    for idx in done_idx:
        if len(ret) > idx+1:
            r = ret[idx+1]
            returns.append(r)
            max_ret = max(r, max_ret)
            min_ret = min(r, min_ret)

    print(min_ret, ' ', max_ret)
    print('Mean: ', np.mean(returns))

    cnt = np.bincount(action, minlength=18)
    prob = cnt/np.sum(cnt)
    print(prob)
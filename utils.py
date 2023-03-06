import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt


def generate_attention_mask(num_obs_tokens, num_non_obs_tokens, seq_len):
    mask_size = (num_obs_tokens + num_non_obs_tokens) * seq_len
    sequential_mask = np.tril(np.ones((mask_size, mask_size)))
    diag = [
        np.ones((num_obs_tokens, num_obs_tokens)) if i % 2 == 0 else np.zeros(
            (num_non_obs_tokens, num_non_obs_tokens))
        for i in range(seq_len * 2)
    ]
    block_diag = scipy.linalg.block_diag(*diag)

    return np.logical_not(np.logical_or(sequential_mask, block_diag))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def plot_return_dist(logits, return_range):
    dist = softmax(logits)
    plt.bar([*range(return_range[0], return_range[1] + 1)], dist)
    plt.show()



def plot_action_dist(logits, n_action):
    dist = softmax(logits)
    plt.bar([str(x) for x in range(0, n_action)], dist)
    plt.show()


def plot_reward_dist(logits):
    dist = softmax(logits)
    plt.bar(['-1', '0', '1'], dist)
    plt.show()

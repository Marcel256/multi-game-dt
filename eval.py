from omegaconf import DictConfig
import hydra
from torch.utils.data import DataLoader

from atari_dataset import AtariDataset
from atari_env.wrapped_env import SequenceEnvironmentWrapper
from model.decision_transformer import DecisionTransformer
from utils import generate_attention_mask
from return_sampling import ReturnSampler, MaxSampler

from torch.distributions import Categorical
import os
import collections
import numpy as np
import scipy
import gym
import imageio
from d4rl_atari.envs import AtariEnv

import torch
import torch.nn as nn


def transform_history(history, device):
    obs = torch.from_numpy(history['observations']).to(device).unsqueeze(0)/255
    action = torch.from_numpy(history['actions']).to(device).unsqueeze(0).long()
    rewards = torch.from_numpy(history['rewards']).to(device).unsqueeze(0).long() + 1

    return obs, action, rewards


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[-1]] = -float('Inf')
    return out



def eval_game(model: DecisionTransformer, mask, env, device, return_sampler: ReturnSampler, record_episode=False):
    context_window = env.reset()
    done = False
    score = 0
    seq_len = model.context_length
    return_stack = collections.deque([99] * seq_len, maxlen=seq_len)
    max_steps = 10000
    step = 0
    frames = []
    ret_dist = []
    a_dist = []

    while not done:
        if record_episode:
            frames.append(context_window['observations'][-1])
        obs, action, rewards = transform_history(context_window, device)

        return_stack.append(0)
        ret = np.stack(return_stack, axis=0)
        ret = torch.from_numpy(ret).to(device).unsqueeze(0)
        return_logits, _, _ = model(obs, ret, action, rewards, attn_mask=mask)
        ret_dist.append(return_logits[0, -1].cpu().detach().numpy())
        sampled_ret = return_sampler.sample_return(return_logits[0,-1])
        ret[0, -1] = sampled_ret
        return_stack[-1] = sampled_ret
        return_logits, action_logits, _ = model(obs, ret, action, rewards, attn_mask=mask)
        a_dist.append(action_logits[0,-1].cpu().detach().numpy())
        dist = Categorical(logits=top_k_logits(action_logits[0, -1], 5))
        sampled_action = dist.sample().item() #torch.argmax(action_logits[0, -1]).item()
        context_window, rew, done, info = env.step(sampled_action)

        score += rew
        step += 1
        if step > max_steps:
            break

    env.close()
    if record_episode:
        with open('eval-obs.npy', 'wb') as fh:
            np.save(file=fh, arr=np.stack(frames, axis=0))

        with open('eval-a-dist.npy', 'wb') as fh:
            np.save(file=fh, arr=np.stack(a_dist, axis=0))

        with open('eval-ret-dist.npy', 'wb') as fh:
            np.save(file=fh, arr=np.stack(ret_dist, axis=0))

    return score, frames


def eval_model_on_games(model: DecisionTransformer, mask, seq_len, games, device, n_runs=10):
    for game in games:
        min, max, mean = eval_model_on_game(model, mask, seq_len, game, device, n_runs=n_runs)
        print('Score for ', game,' Min:', min, ' Max: ', max, ' Mean: ', mean)


def eval_model_on_game(model: DecisionTransformer, mask, seq_len, game, device, n_runs=10):
    model.eval()
    env = AtariEnv(game)
    env = SequenceEnvironmentWrapper(env, seq_len, game_name=game)
    scores = [eval_game(model, mask, env, device, MaxSampler(10))[0] for run in range(n_runs)]
    return np.min(scores), np.max(scores), np.mean(scores)


def eval_model(model: DecisionTransformer, mask, env, device, n_runs=10):
    model.eval()
    scores = [eval_game(model, mask, env, device)[0] for run in range(n_runs)]
    return np.mean(scores)


def eval_model_scores(model: DecisionTransformer, mask, env, device, n_runs=10):
    model.eval()
    for run in range(n_runs):
        print(eval_game(model, mask, env, device)[0])


def record_game(model: DecisionTransformer, mask, seq_len, game, device, file_name='videos/play.gif'):
    model.eval()
    env = AtariEnv(game, clip_reward=True)
    env = SequenceEnvironmentWrapper(env, seq_len, game_name=game)
    score, frames = eval_game(model, mask, env, device, MaxSampler(10), record_episode=True)
    print(score)

    import imageio
    with imageio.get_writer(file_name, mode='I') as writer:
        for frame in frames:
            writer.append_data(frame)


def eval_offline(model: DecisionTransformer, mask, cfg, dataloader, device):
    model.eval()
    return_range = (cfg.model.r_low, cfg.model.r_high)


    returns = 1 + return_range[1] - return_range[0]
    n_actions = cfg.model.n_actions
    n_rewards = cfg.model.n_rewards
    loss = nn.CrossEntropyLoss()
    loss_list = []

    for batch in dataloader:
        obs, ret, action, r = batch

        obs = obs.to(device) / 255
        ret = ret.to(device)
        action = action.to(device).long()
        r = r.to(device)

        ret = torch.clip(ret, return_range[0], return_range[1])
        ret = ret - return_range[0]
        ret = ret.long()

        # 0 for r=-1  1 for r=0 2 for r=1
        r = r.long() + 1
        return_logits, action_logits, reward_logits = model(obs, ret, action, r, attn_mask=mask)

        total_loss = loss(return_logits.view(-1, returns), ret.view(-1)) + loss(action_logits.view(-1, n_actions), action.view(-1)) + loss(reward_logits.view(-1, n_rewards), r.view(-1))
        loss_list.append(total_loss.item())

    return np.mean(loss_list)


def seq_accuracy(model: DecisionTransformer, mask, cfg, dataloader, device):
    model.eval()
    return_range = (cfg.model.r_low, cfg.model.r_high)

    returns = 1 + return_range[1] - return_range[0]
    n_actions = cfg.model.n_actions
    n_rewards = cfg.model.n_rewards

    total_corr_ret = 0
    total_corr_a = 0
    total_corr_r = 0
    n_samples = 0

    for batch in dataloader:
        obs, ret, action, r = batch

        obs = obs.to(device) / 255
        ret = ret.to(device)
        action = action.to(device).long()
        r = r.to(device)

        ret = torch.clip(ret, return_range[0], return_range[1])
        ret = ret - return_range[0]
        ret = ret.long()

        # 0 for r=-1  1 for r=0 2 for r=1
        r = r.long() + 1
        return_logits, action_logits, reward_logits = model(obs, ret, action, r, attn_mask=mask)

        return_pred = torch.argmax(return_logits.view(-1, returns), dim=1)
        action_pred = torch.argmax(action_logits.view(-1, n_actions), dim=1)
        reward_pred = torch.argmax(reward_logits.view(-1, n_rewards), dim=1)

        total_corr_ret += (return_pred == ret.view(-1)).sum().item()
        total_corr_a += (action_pred == action.view(-1)).sum().item()
        total_corr_r = (reward_pred == r.view(-1)).sum().item()
        n_samples += (action.shape[0] * action.shape[1])

    return total_corr_ret/n_samples, total_corr_a/n_samples, total_corr_r/n_samples


def calc_acc(model: DecisionTransformer, mask, cfg, device):
    valid_dataset = AtariDataset('data/valid_breakout', 0, cfg.model.context_length)
    valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.train.batch_size, shuffle=True)

    ret_acc, a_acc, r_acc = seq_accuracy(model, mask, cfg,valid_dataloader, device)
    print(ret_acc)
    print(a_acc)
    print(r_acc)
@hydra.main(version_base=None, config_path="config", config_name="config")
def eval(cfg: DictConfig):


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    seq_len = cfg.model.context_length
    return_range = (cfg.model.r_low, cfg.model.r_high)
    checkpoint = torch.load('models/download/nd-model-27.pt')
    model = DecisionTransformer(cfg.model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    mask = torch.from_numpy(generate_attention_mask(36, 3, seq_len)).to(device)


    eval_model_on_games(model, mask, 4, ['Skiing',
'Breakout',
'DemonAttack',
'SpaceInvaders',
'Assault'], device, n_runs=10)

if __name__ == "__main__":
    eval()
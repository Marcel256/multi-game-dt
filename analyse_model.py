from atari_dataset import AtariDataset

from omegaconf import DictConfig
import hydra
from torch.utils.data import DataLoader


from model.decision_transformer import DecisionTransformer
from utils import generate_attention_mask, plot_action_dist, plot_return_dist, plot_reward_dist

import torch

import matplotlib.pyplot as plt
import seaborn



def action_accuracy(model: DecisionTransformer, mask, cfg, dataloader, device, target_action):
    model.eval()
    return_range = (cfg.model.r_low, cfg.model.r_high)

    returns = 1 + return_range[1] - return_range[0]
    n_actions = cfg.model.n_actions
    n_rewards = cfg.model.n_rewards

    total_corr_ret = 0
    total_corr_a = 0
    total_corr_r = 0
    corr_actions = 0
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

        return_pred = torch.argmax(return_logits[:,-1], dim=1)
        action_pred = torch.argmax(action_logits[:,-1], dim=1)
        reward_pred = torch.argmax(reward_logits[:,-1], dim=1)

        action_mask = action[:,-1] == target_action
        correct = torch.logical_and(action_pred == action[:,-1], action_mask)
        corr_actions += correct.sum()
        n_samples += action_mask.sum()

    print(corr_actions/n_samples)




@hydra.main(version_base=None, config_path="config", config_name="config")
def analyse(cfg: DictConfig):
    valid_dataset = AtariDataset('data/valid_breakout', 0, cfg.model.context_length)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    seq_len = cfg.model.context_length
    return_range = (cfg.model.r_low, cfg.model.r_high)
    checkpoint = torch.load('models/breakout4/model-32.pt')
    model = DecisionTransformer(cfg.model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    model.eval()

    mask = generate_attention_mask(36, 3, seq_len)
    seaborn.heatmap(mask, cmap='rocket_r')
    mask = torch.from_numpy(mask).to(device)

    obs, ret, action, r = next(iter(valid_dataloader))

    obs = obs.to(device) / 255
    n_frames =  obs[0].cpu().numpy()
    for i in range(seq_len):
        plt.imshow(n_frames[i])
        plt.show()
    ret = ret.to(device)
    action = action.to(device).long()
    r = r.to(device)

    ret = torch.clip(ret, return_range[0], return_range[1])
    ret = ret - return_range[0]
    ret = ret.long()
    #ret[0,-1] = 0

    # 0 for r=-1  1 for r=0 2 for r=1
    r = r.long() + 1
    return_logits, action_logits, reward_logits, weights = model.forward_attn(obs, ret, action, r, attn_mask=mask)
    print('Target reward: ', r[0])
    print('Target action: ', action[0])
    print('Target return: ', ret[0])

    print(torch.argmax(action_logits[0], dim=-1))

    seaborn.heatmap(weights[0][0].cpu().detach().numpy(), cmap='rocket_r')


    plot_return_dist(return_logits[0, -1].cpu().detach().numpy(), return_range)
    plot_action_dist(action_logits[0, -1].cpu().detach().numpy(), cfg.model.n_actions)
    plot_reward_dist(reward_logits[0, -1].cpu().detach().numpy())



if __name__ == "__main__":
    analyse()
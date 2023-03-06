from torch.optim.lr_scheduler import ReduceLROnPlateau

import wandb
from omegaconf import DictConfig
import hydra

from model.bctransformer import BCTransformer

import random
import os
import math

import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from atari_dataset import AtariDataset
from utils import generate_attention_mask
#from eval import eval_offline


def eval_offline(model, mask, cfg, valid_dataloader, device):
    model.eval()
    loss = nn.CrossEntropyLoss()
    loss_list =[]
    n_actions = cfg.model.n_actions
    n_rewards = cfg.model.n_rewards
    for batch in valid_dataloader:

        obs, ret, action, r = batch

        obs = obs.to(device) / 255
        action = action.to(device).long()
        r = r.to(device)

        # 0 for r=-1  1 for r=0 2 for r=1
        r = r.long() + 1

        action_logits, reward_logits = model(obs, action, r, attn_mask=mask)

        total_loss = loss(action_logits.view(-1, n_actions), action.view(-1)) + loss(reward_logits.view(-1, n_rewards),
                                                                                     r.view(-1))
        loss_list.append(total_loss.item())

    return np.mean(loss_list)

def lr_decay(step, warmup_steps, final_lr_steps, min_lr_factor):
    if step < warmup_steps:
        # linear lr increase
        lr_factor = step/warmup_steps
    else:
        #cosine lr decay after warmup
        t = float(step - warmup_steps) / float(max(1, final_lr_steps - warmup_steps))
        lr_factor = 0.5 * (1.0 + math.cos(math.pi * t))

    return max(min_lr_factor, lr_factor)


def save_model(model, optimizer, scheduler, path, model_version):
    checkpoint_file = os.path.join(path, 'model-{}.pt'.format(model_version))
    torch.save({
        'model_version': model_version,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, checkpoint_file)

@hydra.main(version_base=None, config_path="config", config_name="config")
def train(cfg: DictConfig):
    wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity)
    wandb.config = {
        "model": cfg.model,
        "train": cfg.train
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()


    seq_len = cfg.model.context_length
    return_range = (cfg.model.r_low, cfg.model.r_high)
    returns = 1 + return_range[1] - return_range[0]
    n_actions = cfg.model.n_actions
    n_rewards = cfg.model.n_rewards

    model = BCTransformer(cfg.model)
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.8, patience=3, min_lr=cfg.train.learning_rate*0.01)#optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: lr_decay(step, cfg.train.warmup_steps,  cfg.train.scheduler_steps,  cfg.train.min_lr_factor))
    model_version = 1

    if 'start_ckpt' in cfg:
        print('Resuming training from ', cfg.start_ckpt)
        checkpoint = torch.load(cfg.start_ckpt)
        model_version = checkpoint['model_version'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    eval_games = ['Asterix',
'BeamRider',
'Breakout',
'DemonAttack',
'Gravitar',
'TimePilot',
'SpaceInvaders',
'Jamesbond',
'Assault',
'Frostbite']



    num_obs_tokens = 36
    num_non_obs_tokens = 2

    dataset_files = 1
    dataset_file_list = [*range(dataset_files)]

    mask = torch.from_numpy(generate_attention_mask(num_obs_tokens, num_non_obs_tokens, seq_len)).to(device)

    loss = nn.CrossEntropyLoss()

    loss_list = []
    i = 0
    log_loss_steps = 100
    validation_steps = 500
    valid_dataset = AtariDataset('data/valid_breakout', 0, cfg.model.context_length)
    valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.train.batch_size, shuffle=True)

    model.train()

    for epoch in range(cfg.train.epochs):
        random.shuffle(dataset_file_list)
        for dataset_file in dataset_file_list:
            dataset = AtariDataset('data/single_split/Breakout', dataset_file, cfg.model.context_length)
            dataloader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True)
            for batch in tqdm(dataloader):
                optimizer.zero_grad()
                obs, ret, action, r = batch

                obs = obs.to(device) / 255
                action = action.to(device).long()
                r = r.to(device)

                # 0 for r=-1  1 for r=0 2 for r=1
                r = r.long() + 1

                action_logits, reward_logits = model(obs, action, r, attn_mask=mask)

                total_loss =  loss(action_logits.view(-1, n_actions), action.view(-1)) #+ loss(reward_logits.view(-1, n_rewards), r.view(-1))
                total_loss.backward()
                loss_list.append(total_loss.item())
                if (i+1) % log_loss_steps == 0:
                    train_loss = np.mean(loss_list)
                    wandb.log({"train_loss": train_loss}) # "learning_rate": scheduler.get_last_lr()[0]
                    loss_list = []

                if (i+1) % validation_steps == 0:
                    valid_loss = eval_offline(model, mask, cfg, valid_dataloader, device)
                    wandb.log({"valid_loss": valid_loss})
                    model.train()
                    save_model(model, optimizer, scheduler, cfg.ckpt_path, model_version)
                    model_version += 1
                    scheduler.step(valid_loss)

                i += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()

            del dataloader
            del dataset

    save_model(model, optimizer, scheduler, cfg.ckpt_path, model_version)


if __name__ == "__main__":
    train()
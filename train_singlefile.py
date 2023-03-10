import wandb
from omegaconf import DictConfig
import hydra

from model.decision_transformer import DecisionTransformer

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

import gc
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = "1"


def lr_decay(step, warmup_steps, final_lr_steps, min_lr_factor):
    if step < warmup_steps:
        # linear lr increase
        lr_factor = step / warmup_steps
    else:
        # cosine lr decay after warmup
        # t = float(step - warmup_steps) / float(max(1, final_lr_steps - warmup_steps))
        lr_factor = 1  # 0.5 * (1.0 + math.cos(math.pi * t))

    return max(min_lr_factor, lr_factor)


def save_model(model, optimizer, scheduler, cfg, model_version):
    checkpoint_file = os.path.join(cfg.ckpt_path, 'model-{}.pt'.format(model_version))
    torch.save({
        'model_version': model_version,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, checkpoint_file)
    artifact = wandb.Artifact(name=cfg.wandb.model_name, type='model', metadata={
        "version": model_version,
    })
    artifact.add_file(checkpoint_file)
    wandb.log_artifact(artifact)


@hydra.main(version_base=None, config_path="config", config_name="config")
def train(cfg: DictConfig):
    gc.collect()
    torch.cuda.empty_cache()
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

    model = DecisionTransformer(cfg.model)
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: lr_decay(step, cfg.train.warmup_steps,
                                                                                       cfg.train.scheduler_steps,
                                                                                       cfg.train.min_lr_factor))
    model_version = 1

    if 'start_ckpt' in cfg:
        print('Resuming training from ', cfg.start_ckpt)
        checkpoint = torch.load(cfg.start_ckpt)
        model_version = checkpoint['model_version'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    num_obs_tokens = 36
    num_non_obs_tokens = 3

    dataset_files = 1
    dataset_file_list = [*range(dataset_files)]

    mask = torch.from_numpy(generate_attention_mask(num_obs_tokens, num_non_obs_tokens, seq_len)).to(device)

    loss = nn.CrossEntropyLoss()

    loss_list = []
    i = 0
    log_loss_steps = 100
    validation_steps = 5000

    model.train()
    dataset = AtariDataset('data/merged2', 0, cfg.model.context_length)
    dataloader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True)

    for epoch in range(cfg.train.epochs):
        for batch in tqdm(dataloader):
            optimizer.zero_grad()
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

            total_loss = loss(return_logits.view(-1, returns), ret.view(-1)) + loss(action_logits.view(-1, n_actions),
                                                                                    action.view(-1)) + loss(
                reward_logits.view(-1, n_rewards), r.view(-1))
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            scheduler.step()
            loss_list.append(total_loss.item())
            if (i + 1) % log_loss_steps == 0:
                train_loss = np.mean(loss_list)
                wandb.log({"train_loss": train_loss})  # "learning_rate": scheduler.get_last_lr()[0]
                loss_list = []

            if (i + 1) % validation_steps == 0:
                # valid_loss = eval_offline(model, mask, cfg, valid_dataloader, device)
                # wandb.log({"valid_loss": valid_loss})
                # model.train()
                save_model(model, optimizer, scheduler, cfg, model_version)
                model_version += 1

            i += 1

    save_model(model, optimizer, scheduler, cfg.ckpt_path, model_version)


if __name__ == "__main__":
    train()

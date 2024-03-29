import torch.nn as nn
import torch
from model.transformer import Transformer


class DecisionTransformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.emb_dim = config.emb_dim
        self.context_length = config.context_length
        returns = 1 + config.r_high - config.r_low
        n_rewards = config.n_rewards
        n_actions = config.n_actions
        n_obs_token = config.n_obs_token
        self.return_range = (config.r_low, config.r_high)
        self.transformer = Transformer(self.emb_dim, config.n_heads, config.n_layers)
        self.state_encoder = nn.Conv2d(1, self.emb_dim, config.obs_patch_size, config.obs_patch_size)
        self.return_emb = nn.Embedding(returns, self.emb_dim)
        self.action_emb = nn.Embedding(n_actions, self.emb_dim)
        self.reward_emb = nn.Embedding(n_rewards, self.emb_dim)
        self.emb_drop = nn.Dropout(0.1)

        self.return_out = nn.Linear(self.emb_dim, returns, bias=False)
        self.action_out = nn.Linear(self.emb_dim, n_actions, bias=False)
        self.reward_out = nn.Linear(self.emb_dim, n_rewards, bias=False)

        self.img_pos_emb = torch.nn.Parameter(torch.randn((self.emb_dim, config.n_obs_token))*0.02)
        self.img_pos_emb.requires_grad = True
        self.pos_emb = torch.nn.Parameter(torch.randn((config.context_length*(n_obs_token+3), self.emb_dim))*0.02)
        self.pos_emb.requires_grad = True

        self.out_ln = nn.LayerNorm(self.emb_dim)

    def encode_obs(self, x):
        batch_size = x.shape[0]
        seq_length = x.shape[1]
        #[B, T, 84, 84]
        images = torch.reshape(x, (batch_size * seq_length, 1, x.shape[2], x.shape[3]))
        # [B*T, 1, W, H]
        emb = self.state_encoder(images)
        # [B*T, D, P, P]
        emb = torch.reshape(emb, (batch_size, seq_length, emb.shape[1], emb.shape[2] * emb.shape[3]))
        # [B, T, D, P*P]

        # add positional encoding for obs token
        emb = emb + self.img_pos_emb

        emb = torch.reshape(emb, (batch_size, seq_length, -1))
        # [B, T, D*P*P]
        return emb

    def token_dropout(self, token):
        if self.training:
            device = token.device
            mask = torch.rand((token.shape[0], token.shape[1], 1), device=device) > 0.25
            return token * mask

        return token

    def encode_ret(self, ret):
        return self.token_dropout(self.return_emb(ret))

    def encode_actions(self, actions):
        return self.token_dropout(self.action_emb(actions))

    def encode_rewards(self, rewards):
        return self.token_dropout(self.reward_emb(rewards))



    def forward(self, obs, returns, actions, rewards, attn_mask=None):
        batch_size = actions.shape[0]
        seq_length = actions.shape[1]
        obs_token = self.encode_obs(obs)
        ret_token = self.encode_ret(returns)
        act_token = self.encode_actions(actions)
        r_token = self.encode_rewards(rewards)

        token = torch.cat((obs_token, ret_token, act_token, r_token), dim=-1)
        token = torch.reshape(token, (batch_size, -1, self.emb_dim))
        token = token + self.pos_emb
        token_per_timestep = int(token.shape[1] / seq_length)
        num_obs_token = token_per_timestep-3
        out = self.out_ln(self.transformer(token, attn_mask))
        return_logits = self.return_out(out[:,num_obs_token-1::token_per_timestep])
        action_logits = self.action_out(out[:,num_obs_token::token_per_timestep])
        reward_logits = self.reward_out(out[:,num_obs_token+1::token_per_timestep])

        return return_logits, action_logits, reward_logits


    def forward_attn(self, obs, returns, actions, rewards, attn_mask=None):
        batch_size = actions.shape[0]
        seq_length = actions.shape[1]
        obs_token = self.encode_obs(obs)
        ret_token = self.encode_ret(returns)
        act_token = self.encode_actions(actions)
        r_token = self.encode_rewards(rewards)

        token = torch.cat((obs_token, ret_token, act_token, r_token), dim=-1)
        token = torch.reshape(token, (batch_size, -1, self.emb_dim))
        token = token + self.pos_emb
        token_per_timestep = int(token.shape[1] / seq_length)
        num_obs_token = token_per_timestep-3
        transformer_out, weights = self.transformer.forward_attn(token, attn_mask)
        out = self.out_ln(transformer_out)
        return_logits = self.return_out(out[:,num_obs_token-1::token_per_timestep])
        action_logits = self.action_out(out[:,num_obs_token::token_per_timestep])
        reward_logits = self.reward_out(out[:,num_obs_token+1::token_per_timestep])

        return return_logits, action_logits, reward_logits, weights
import torch
from torch.distributions import Categorical


class ReturnSampler(object):

    def sample_return(self, logits):
        pass


class MaxSampler(ReturnSampler):

    def __init__(self, n):
        self.n = n

    def sample_return(self, logits):
        dist = Categorical(logits=logits)
        return dist.sample((self.n,)).max().item()


class ExpertSample(ReturnSampler):

    def __init__(self, return_range, kappa):
        returns = 1+return_range[1]-return_range[0]
        self.expert_actions = kappa * torch.arange(returns)/returns

    def sample_return(self, logits):
        dist = Categorical(logits=(logits+self.expert_actions))
        return dist.sample().item()
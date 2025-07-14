import sys
import os
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
sys.path.append(ROOT)

from ..builder import AGENTS
from ..custom import AgentBase
from trademaster.utils import get_attr
import torch
from torch.distributions import Normal
import random
import pandas as pd
import numpy as np
from torch.distributions import Categorical

@AGENTS.register_module()
class PortfolioManagementInvestorImitator(AgentBase):
    def __init__(self, **kwargs):
        super(PortfolioManagementInvestorImitator, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.act = get_attr(kwargs, "act", None).to(self.device)

        self.act_optimizer = get_attr(kwargs, "act_optimizer", None)
        self.criterion = get_attr(kwargs, "criterion", None)
        self.memory_counter = get_attr(kwargs, "memory_counter", None)

        self.action_dim = get_attr(kwargs, "action_dim", None)
        self.state_dim = get_attr(kwargs, "state_dim", None)

    def get_save(self):
        models = {
            "act":self.act,
        }
        optimizers = {
            "act_optimizer":self.act_optimizer,
        }
        res = {
            "models":models,
            "optimizers":optimizers
        }
        return res

    def get_action(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        
        # DEBUG: Check for NaN/inf in input state
        print(f"DEBUG: State shape: {state.shape}")
        print(f"DEBUG: State min/max: {state.min():.6f}/{state.max():.6f}")
        print(f"DEBUG: State values: {state.flatten()[:10]}")  # First 10 values
        
        if torch.isnan(state).any() or torch.isinf(state).any():
            print(f"ERROR: State contains NaN/inf: {state}")
            exit(1)
            
        # DEBUG: Check network weights
        for name, param in self.act.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"ERROR: Network weights contain NaN/inf in {name}")
                exit(1)
        
        logits = self.act(state)
        
        # DEBUG: Check for NaN/inf in network output
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"ERROR: Network output contains NaN/inf: {logits}")
            print(f"Logits shape: {logits.shape}")
            exit(1)
            
        probs = torch.softmax(logits, dim=-1)
        
        # DEBUG: Check for NaN/inf in probabilities
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            print(f"ERROR: Probabilities contain NaN/inf: {probs}")
            print(f"Original logits: {logits}")
            exit(1)
            
        m = Categorical(probs)
        action = m.sample()
        self.act.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def update_net(self):
        R = 0
        policy_loss = []
        returns = []
        for r in self.act.rewards[::-1]:
            # R = r + R * self.gamma
            R = r
            returns.insert(0, R)
        returns = torch.tensor(returns)
        for log_prob, R in zip(self.act.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        self.act_optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()  # 求和
        policy_loss.backward()
        self.act_optimizer.step()
        del self.act.rewards[:]  # 清空episode 数据
        del self.act.saved_log_probs[:]
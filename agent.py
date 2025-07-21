#!/usr/bin/env python
# coding: utf-8

# In[10]:


# agent.py

"""
DDPG Agent
"""

from typing import List
import numpy as np
import torch
import torch.nn.functional as functional
from torch import optim, nn
import model as model
import utils as utils
from config import Config


class DDPGAgent(object):
    """
    DDPG (Deep Deterministic Policy Gradient) Agent
    """
    def __init__(
        self,
        config: Config,
        noise: utils.OUNoise,
        group2members_dict: dict,
        visual_features,
        textual_features,
        verbose=False
    ):
        self.config = config
        self.noise = noise
        self.group2members_dict = group2members_dict
        self.tau = config.tau
        self.gamma = config.gamma
        self.device = config.device

        
        self.embedding = model.Embedding(
            embedding_size=config.embedding_size,
            user_num=config.user_num,
            item_num=config.item_num,
            num_heads=config.num_heads,
            dropout_rate=config.dropout_rate,
            visual_feat_dim=config.visual_feature_dim,
            textual_feat_dim=config.textual_feature_dim,
            visual_features=visual_features,
            textual_features=textual_features
        ).to(config.device)

        self.actor = model.Actor(
            embedded_state_size=config.embedded_state_size,
            action_weight_size=config.embedded_action_size,
            hidden_sizes=config.actor_hidden_sizes
        ).to(config.device)

        
        self.actor_target = model.Actor(config.embedded_state_size, config.embedded_action_size, config.actor_hidden_sizes).to(config.device)
        self.critic = model.Critic(config.embedded_state_size, config.embedded_action_size, config.critic_hidden_sizes).to(config.device)
        self.critic_target = model.Critic(config.embedded_state_size, config.embedded_action_size, config.critic_hidden_sizes).to(config.device)

        if verbose:
            print(self.embedding)
            print(self.actor)
            print(self.critic)

        self.copy_network(self.actor, self.actor_target)
        self.copy_network(self.critic, self.critic_target)

        self.replay_memory = utils.ReplayMemory(buffer_size=config.buffer_size)
        self.critic_criterion = nn.MSELoss()
        self.embedding_optimizer = optim.Adam(self.embedding.parameters(), lr=config.embedding_learning_rate, weight_decay=config.embedding_weight_decay)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_learning_rate)
        self.actor_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=100, gamma=0.9)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.actor_learning_rate)
        self.critic_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=100, gamma=0.9)


    def copy_network(self, network: nn.Module, network_target: nn.Module):
        """
        Copy one network to its target network

        :param network: the original network to be copied
        :param network_target: the target network
        """
        for parameters, target_parameters in zip(network.parameters(), network_target.parameters()):
            target_parameters.data.copy_(parameters.data)

    def sync_network(self, network: nn.Module, network_target: nn.Module):
        """
        Synchronize one network to its target network

        :param network: the original network to be synchronized
        :param network_target: the target network
        :return:
        """
        for parameters, target_parameters in zip(network.parameters(), network_target.parameters()):
            target_parameters.data.copy_(parameters.data * self.tau + target_parameters.data * (1 - self.tau))

    def get_action(self, state: list, item_candidates: list = None, top_K: int = 1, with_noise=False):
        """
        Get one action

        :param state: one environment state
        :param item_candidates: item candidates
        :param top_K: top K items
        :param with_noise: True to with noise
        :return: action
        """
        with torch.no_grad():
            states = [state]
            embedded_states = self.embed_states(states)
            action_weights = self.actor(embedded_states)
            action_weight = torch.squeeze(action_weights)
            if with_noise:
                action_weight += self.noise.get_ou_noise()

            if item_candidates is None:
                item_embedding_weight = self.embedding.item_embedding.weight.clone()
            else:
                item_candidates = np.array(item_candidates)
                item_candidates_tensor = torch.tensor(item_candidates, dtype=torch.int).to(self.device)
                item_embedding_weight = self.embedding.item_embedding(item_candidates_tensor)

            scores = torch.inner(action_weight, item_embedding_weight).detach().cpu().numpy()
            sorted_score_indices = np.argsort(scores)[:top_K]

            if item_candidates is None:
                action = sorted_score_indices
            else:
                action = item_candidates[sorted_score_indices]
            action = np.squeeze(action)
            if top_K == 1:
                action = action.item()
        return action

    def get_embedded_actions(self, embedded_states: torch.Tensor, target=False):
        """
        Get embedded actions

        :param embedded_states: embedded states
        :param target: True for target network
        :return: embedded_actions (, actions)
        """
        if not target:
            action_weights = self.actor(embedded_states)
        else:
            action_weights = self.actor_target(embedded_states)

        item_embedding_weight = self.embedding.item_embedding.weight.clone()
        scores = torch.inner(action_weights, item_embedding_weight)
        embedded_actions = torch.inner(functional.gumbel_softmax(scores, hard=True), item_embedding_weight.t())
        
        return embedded_actions

    def embed_state(self, state: list):
        """
        Embed one state

        :param state: state
        :return: embedded_state
        """
        group_id = state[0]
        group_members = torch.tensor(self.group2members_dict[group_id], dtype=torch.int).to(self.device)
        
        history = torch.tensor(state[1:], dtype=torch.int).to(self.device)
        embedded_state = self.embedding(group_members, history)

        return embedded_state

    def embed_states(self, states):
        embedded_states = [self.embed_state(state) for state in states]
        embedded_states = torch.stack(embedded_states, dim=0)

        # Flatten or reshape if necessary to match the size of embedded_actions
        if embedded_states.dim() > 2:
            # Assuming we want to flatten the last two dimensions
            embedded_states = embedded_states.view(-1, embedded_states.size(-1) * embedded_states.size(-2))

        return embedded_states

    def embed_actions(self, actions):
        actions = torch.tensor(actions, dtype=torch.int).to(self.device)
        embedded_actions = self.embedding.item_embedding(actions)

        # Check and adjust the shape of embedded_actions to match embedded_states
        if embedded_actions.dim() > 2:
            # Flatten if necessary
            embedded_actions = embedded_actions.view(-1, embedded_actions.size(-1) * embedded_actions.size(-2))

        return embedded_actions

    def update(self):
        """
        Update the networks

        :return: actor loss and critic loss
        """
        batch = self.replay_memory.sample(self.config.batch_size)
        states, actions, rewards, next_states = list(zip(*batch))

        self.embedding_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        embedded_states = self.embed_states(states)
        embedded_actions = self.embed_actions(actions)
        rewards = torch.unsqueeze(torch.tensor(rewards, dtype=torch.int).to(self.device), dim=-1)
        embedded_next_states = self.embed_states(next_states)
        q_values = self.critic(embedded_states, embedded_actions)

        with torch.no_grad():
            embedded_next_actions = self.get_embedded_actions(embedded_next_states, target=True)
            next_q_values = self.critic_target(embedded_next_states, embedded_next_actions)
            q_values_target = rewards + self.gamma * next_q_values

        critic_loss = self.critic_criterion(q_values, q_values_target)
        critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        embedded_states = self.embed_states(states)
        actor_loss = -self.critic(embedded_states, self.get_embedded_actions(embedded_states)).mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor_scheduler.step()
        self.critic_scheduler.step()
        
        self.embedding_optimizer.step()
        
        self.sync_network(self.actor, self.actor_target)
        self.sync_network(self.critic, self.critic_target)

        return actor_loss.detach().cpu().numpy(), critic_loss.detach().cpu().numpy()


if __name__ == '__main__':
    pass


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# main.py

"""
Main
"""
import pandas as pd
from agent import DDPGAgent
from config import Config
from data import DataLoader
from env import Env
from eval import Evaluator
from utils import OUNoise
import os
import numpy as np
import torch

def train(config: Config, env: Env, agent: DDPGAgent, evaluator: Evaluator,
          df_eval_user: pd.DataFrame, df_eval_group: pd.DataFrame):
    """
    Train the agent with the environment
    """
    rewards = []
    for episode in range(config.num_episodes):
        state = env.reset()
        agent.noise.reset()
        episode_reward = 0

        for step in range(config.num_steps):
            action = agent.get_action(state)
            new_state, reward, _, _ = env.step(action)
            agent.replay_memory.push((state, action, reward, new_state))
            state = new_state
            episode_reward += reward

            if len(agent.replay_memory) >= config.batch_size:
                agent.update()

        rewards.append(episode_reward / config.num_steps)
        print('Episode = %d, average reward = %.4f' % (episode, episode_reward / config.num_steps))

        # Evaluate periodically
        if (episode + 1) % config.eval_per_iter == 0:
            for top_K in config.top_K_list:
                evaluator.evaluate(agent=agent, df_eval=df_eval_user, mode='user', top_K=top_K)
            for top_K in config.top_K_list:
                evaluator.evaluate(agent=agent, df_eval=df_eval_group, mode='group', top_K=top_K)

if __name__ == '__main__':
    config = Config()
    dataloader = DataLoader(config)

    # Use 'train' for training, 'val' for validation, 'test' for final test
    rating_matrix_train = dataloader.load_rating_matrix(dataset_name='train')
    df_eval_user_val = dataloader.load_eval_data(mode='user', dataset_name='val')
    df_eval_group_val = dataloader.load_eval_data(mode='group', dataset_name='val')

    env = Env(config=config, rating_matrix=rating_matrix_train, dataset_name='train')
    noise = OUNoise(config=config)

    agent = DDPGAgent(
        config=config,
        noise=noise,
        group2members_dict=dataloader.group2members_dict,
        visual_features=dataloader.movie_posters,
        textual_features=dataloader.movie_textual_features,
        verbose=True
    )

    evaluator = Evaluator(config=config)

    # run an initial evaluation
    print("Initial evaluation (before training):")
    evaluator.evaluate(agent, df_eval_user_val, mode='user', top_K=5)
    evaluator.evaluate(agent, df_eval_group_val, mode='group', top_K=5)

    #Train
    train(config=config, env=env, agent=agent, evaluator=evaluator,
          df_eval_user=df_eval_user_val, df_eval_group=df_eval_group_val)

    # run a final evaluation
    print("Final evaluation (after training):")
    evaluator.evaluate(agent, df_eval_user_val, mode='user', top_K=5)
    evaluator.evaluate(agent, df_eval_group_val, mode='group', top_K=5)


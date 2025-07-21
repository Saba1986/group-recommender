#!/usr/bin/env python
# coding: utf-8

# In[5]:


# config.py

"""
Configurations
"""
import os
import torch


class Config(object):
    """
    Configurations
    """
    def __init__(self, use_csv=True):
        # Set to True if the MovieLens data is in CSV format
        self.use_csv = use_csv

        self.data_folder_path = os.path.join('..', 'data', 'MovieLens-1M')
        
        
        # Choose file extensions based on format
        if self.use_csv:
            self.item_path = os.path.join(self.data_folder_path, 'movies.csv')
            self.user_path = os.path.join(self.data_folder_path, 'users.csv')
        else:
            self.item_path = os.path.join(self.data_folder_path, 'movies.dat')
            self.user_path = os.path.join(self.data_folder_path, 'users.dat')

        self.group_path = os.path.join(self.data_folder_path, 'groupMember.dat')
        self.saves_folder_path = os.path.join('saves')
        self.num_heads = 16

        self.dropout_rate = 0.2  # Define the dropout rate for the MultiHeadAttention

#        # Paths to precomputed visual and textual features
        self.poster_folder = os.path.join(self.data_folder_path, 'posters')  # folder with .jpg posters
        self.movie_posters_npy = os.path.join(self.data_folder_path, 'movie_posters.npy')
        self.movie_textual_features_npy = os.path.join(self.data_folder_path, 'movie_textual_features.npy')

        
        # Feature dimensions
        self.visual_feature_dim = 1280  # e.g., EfficientNet-B0 output
        self.textual_feature_dim = 300  # e.g., GloVe embeddings

        # Recommendation system
        self.history_length = 5
        self.top_K_list = [5, 10, 15]
        self.rewards = [0, 1]

        # Reinforcement learning
        self.embedding_size = 32
        self.state_size = self.history_length + 1
        self.action_size = 1
        self.embedded_action_size = self.action_size * self.embedding_size
        
        self.embedded_state_size = (
            self.embedding_size + 
            self.history_length * (self.embedding_size + self.visual_feature_dim + self.textual_feature_dim)
        )

        # Numbers
        self.item_num = None
        self.user_num = None
        self.group_num = None
        self.total_group_num = None

        # Environment
        self.env_n_components = self.embedding_size
        self.env_tol = 1e-4
        self.env_max_iter = 1000
        self.env_alpha = 0.001

        # Actor-Critic network
        self.actor_hidden_sizes = (128, 64, 32)
        self.critic_hidden_sizes = (32, 16)

        # DDPG algorithm
        self.tau = 1e-3
        self.gamma = 0.9

        # Optimizer
        self.batch_size = 64
        self.buffer_size = 100000
        self.num_episodes = 1000
        self.num_steps = 100
        self.embedding_weight_decay = 1e-6
        self.actor_weight_decay = 1e-6
        self.critic_weight_decay = 1e-6
        self.embedding_learning_rate = 0.0001
        self.actor_learning_rate = 0.0001
        self.critic_learning_rate = 0.0001
        self.eval_per_iter = 10

        # OU noise
        self.ou_mu = 0.0
        self.ou_theta = 0.15
        self.ou_sigma = 0.2
        self.ou_epsilon = 1.0

        # GPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")


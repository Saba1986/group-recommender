#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# model.py

"""
Models
"""
from typing import Tuple
import torch
import torch.nn as nn
from config import Config

class Actor(nn.Module):
    """
    Actor Network
    """
    def __init__(self, embedded_state_size: int, action_weight_size: int, hidden_sizes: Tuple[int]):
        """
        Initialize Actor
        :param embedded_state_size: embedded state size
        :param action_weight_size: embedded action size
        :param hidden_sizes: hidden sizes
        """        
        super(Actor, self).__init__()
        config = Config()
        self.dropout = nn.Dropout(p=config.dropout_rate)
        self.net = nn.Sequential(
            nn.Linear(embedded_state_size, hidden_sizes[0]),
            nn.ReLU(),
            self.dropout,
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            # Adding another layer
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[2], action_weight_size)
        )

    def forward(self, embedded_state):
        """
        Forward
        :param embedded_state: embedded state
        :return: action weight
        """
        return self.net(embedded_state)


class Critic(nn.Module):
    """
    Critic Network
    """
    def __init__(self, embedded_state_size: int, embedded_action_size: int, hidden_sizes: Tuple[int]):
        """
        Initialize Critic

        :param embedded_state_size: embedded state size
        :param embedded_action_size: embedded action size
        :param hidden_sizes: hidden sizes
        """
        super(Critic, self).__init__()
        config = Config()
        self.dropout = nn.Dropout(p=config.dropout_rate)
        total_input_size = embedded_state_size + embedded_action_size    

        self.net = nn.Sequential(
            nn.Linear(total_input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            # Adding another layer
            nn.Linear(hidden_sizes[1], 1)
        )

    def forward(self, embedded_state, embedded_action):
        """
        Forward
        :param embedded_state: embedded state
        :param embedded_action: embedded action
        :return: Q value
        """
        if embedded_state.dim() != embedded_action.dim():
            raise ValueError("embedded_state and embedded_action must have the same number of dimensions")

        combined_input = torch.cat([embedded_state, embedded_action], dim=-1)
        return self.net(combined_input)
    
    
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module
    """
    def __init__(self, embedding_size: int, num_heads: int, dropout_rate: float):
        """
        Initialize Multi-Head Attention
        :param embedding_size: embedding size
        :param num_heads: number of attention heads
        :param dropout_rate: dropout rate
        """
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_size = embedding_size // num_heads

        self.query_projection = nn.Linear(embedding_size, embedding_size)
        self.key_projection = nn.Linear(embedding_size, embedding_size)
        self.value_projection = nn.Linear(embedding_size, embedding_size)
        self.output_projection = nn.Linear(embedding_size, embedding_size)

        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, queries, keys, values):
        """
        Forward
        :param queries: query tensor
        :param keys: key tensor
        :param values: value tensor
        :return: output tensor
        """
        batch_size = queries.size(0)

        # Project queries, keys, and values
        projected_queries = self.query_projection(queries)
        projected_keys = self.key_projection(keys)
        projected_values = self.value_projection(values)

        # Reshape projected tensors for multi-head attention
        projected_queries = projected_queries.view(batch_size, -1, self.num_heads, self.head_size)
        projected_keys = projected_keys.view(batch_size, -1, self.num_heads, self.head_size)
        projected_values = projected_values.view(batch_size, -1, self.num_heads, self.head_size)

        # Transpose dimensions for matrix multiplication
        projected_queries = projected_queries.transpose(1, 2)
        projected_keys = projected_keys.transpose(1, 2)
        projected_values = projected_values.transpose(1, 2)

        # Compute attention scores
        attention_scores = torch.matmul(projected_queries, projected_keys.transpose(-2, -1))
        attention_scores = attention_scores / (self.head_size ** 0.5)

        # Apply softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Apply dropout
        attention_weights = self.dropout(attention_weights)

        # Apply attention weights to values
        attention_output = torch.matmul(attention_weights, projected_values)

        # Transpose and reshape attention output
        attention_output = attention_output.transpose(1, 2)
        attention_output = attention_output.contiguous().view(batch_size, -1, self.num_heads * self.head_size)

        # Project attention output
        output = self.output_projection(attention_output)
        return output     
    
class Embedding(nn.Module):
    def __init__(self, embedding_size, user_num, item_num, num_heads, dropout_rate,
                 visual_feat_dim, textual_feat_dim, visual_features, textual_features):
        super().__init__()
        print(f"DEBUG: user_num={user_num} ({type(user_num)}), item_num={item_num} ({type(item_num)})")
        user_num = int(user_num)
        item_num = int(item_num)
        self.user_embedding = nn.Embedding(user_num + 1, embedding_size)
        self.item_embedding = nn.Embedding(item_num + 1, embedding_size)
        self.visual_features = torch.tensor(visual_features, dtype=torch.float32) 
        self.textual_features = torch.tensor(textual_features, dtype=torch.float32)
        self.user_multihead_attention = MultiHeadAttention(embedding_size, num_heads, dropout_rate)
        self.visual_feat_dim = visual_feat_dim
        self.textual_feat_dim = textual_feat_dim
        
    def forward(self, group_members, history):
        # Group embedding via Multi-Head Attention
        embedded_group_members = self.user_embedding(group_members)  # (group_size, embedding_size)
        # Add batch dimension (1, group_size, embedding_size)
        embedded_group_members = embedded_group_members.unsqueeze(0)

        # Handle different group sizes
        # Use weighted average aggregation strategy over smaller groups and attention over larger groups
        group_size = group_members.size(-1)
        if group_size <= 3:
            weights = torch.softmax(torch.norm(embedded_group_members, dim=-1), dim=-1)
            embedded_group = torch.sum(weights.unsqueeze(-1) * embedded_group_members, dim=1)
        else:
            group_attn_output = self.user_multihead_attention(
                queries=embedded_group_members,
                keys=embedded_group_members,
                values=embedded_group_members
            )
            embedded_group = group_attn_output.mean(dim=1)   
        
        # Multimodal History
        embedded_history = self.item_embedding(history)  # (history_length, embedding_size)
        history_visual = self.visual_features[history]   # (history_length, visual_feat_dim)
        history_textual = self.textual_features[history] # (history_length, textual_feat_dim)
        multimodal_history = torch.cat([embedded_history, history_visual, history_textual], dim=-1)
        multimodal_history_flat = multimodal_history.view(1, -1)
#        multimodal_history_flat = multimodal_history.view(multimodal_history.size(0), -1)

        # Combine
        embedded_state = torch.cat([embedded_group, multimodal_history_flat], dim=-1)
        return embedded_state


# In[ ]:





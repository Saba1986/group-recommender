Overview

MM-GRDRL is a state-of-the-art group recommendation system designed to dynamically generate personalized recommendations for groups by integrating visual, textual, and behavioral user data through deep reinforcement learning (DRL). 
The framework employs an adaptive aggregation strategy, combining weighted averaging for smaller groups and multi-head attention for larger groups.


Features

Multi-Modal Fusion: Integrates visual features (EfficientNet-B0) from movie posters, textual features (GloVe embeddings) from movie titles, and behavioral signals from user-item interactions.

Adaptive Aggregation: Implements weighted average aggregation for smaller groups and multi-head attention for larger groups.

Deep Reinforcement Learning: Uses the actor-critic architecture within a Markov Decision Process (MDP) formulation to adaptively refine recommendations based on group feedback.


Repository Structure

agent.py: Contains the DRL agent logic.

config.py: Configuration parameters and hyperparameters.

data.py: Data preprocessing and handling utilities.

env.py: Simulation environment.

eval.py: Evaluation scripts.

generator.py: Generates group interaction scenarios.

model.py: Neural network architectures including multi-head attention and embedding layers.

utils.py: Utility functions for data handling and evaluation.

main.py: Entry point for training and evaluating the model.

Visual-Textual Features.py: Scripts for extracting visual and textual features.

MovieLens-Rand/: Contains the processed dataset and relevant files.


Dataset

The MovieLens-Rand dataset is derived from MovieLens 1M, containing generated group interactions suitable for group recommendation scenarios.



Running the Code

To train and evaluate the MM-GRDRL model, run:

python main.py



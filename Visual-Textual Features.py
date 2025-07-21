#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Visual-Textual Features.py

from torchvision import models, transforms
from PIL import Image
import torch

# Load pretrained model
model = models.efficientnet_b0(pretrained=True)
model.eval()

# Transform for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def extract_poster_feature(img_path):
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = model.features(img)
        pooled = features.mean(dim=[2, 3])  # Global Average Pooling
    return pooled.squeeze().numpy()


# In[2]:


from torchtext.vocab import GloVe

glove = GloVe(name='6B', dim=300)
def title_to_vector(title):
    tokens = title.lower().replace('(', '').replace(')', '').replace(':', '').split()
    vectors = [glove[token].numpy() for token in tokens if token in glove.stoi]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(glove.dim)


# In[5]:


# A. Download Posters

import pandas as pd
import requests
import os
movies = pd.read_csv('../data/MovieLens-Rand/movies.csv')
poster_dir = '../data/MovieLens-Rand/posters'
os.makedirs(poster_dir, exist_ok=True)

for idx, row in movies.iterrows():
    url = row['poster_url']
    movie_id = row['movie_id']
    save_path = os.path.join(poster_dir, f"{movie_id}.jpg")
    if not os.path.exists(save_path):
        try:
            resp = requests.get(url, timeout=5)
            with open(save_path, 'wb') as f:
                f.write(resp.content)
        except Exception as e:
            print(f"Failed to download {url}: {e}")


# In[6]:


# B. Extract Visual Features

import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import os
from tqdm import tqdm

poster_dir = '../data/MovieLens-Rand/posters'
movies = pd.read_csv('../data/MovieLens-Rand/movies.csv')
movie_ids = movies['movie_id'].tolist()
feature_dim = 1280  # EfficientNetB0 output

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.efficientnet_b0(pretrained=True).to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

features = np.zeros((max(movie_ids) + 1, feature_dim))

for movie_id in tqdm(movie_ids):
    path = os.path.join(poster_dir, f"{movie_id}.jpg")
    if not os.path.exists(path):
        continue
    img = Image.open(path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.features(img)
        pooled = feat.mean(dim=[2, 3]).squeeze().cpu().numpy()
        features[movie_id] = pooled

np.save('movie_posters.npy', features)


# In[2]:


from torchvision import models, transforms
from PIL import Image
import torch


# In[3]:


# Create movie_textual_features.npy (Textual Features)
# Using GloVe Embeddings

import numpy as np
import pandas as pd
from torchtext.vocab import GloVe

movies = pd.read_csv('../data/MovieLens-Rand/movies.csv')
movie_ids = movies['movie_id'].tolist()
glove = GloVe(name='6B', dim=300)
features = np.zeros((max(movie_ids) + 1, 300))

def text_to_vec(text):
    tokens = str(text).lower().replace('(', '').replace(')', '').replace(':', '').split()
    vecs = [glove[t] for t in tokens if t in glove.stoi]
    if vecs:
        return torch.stack(vecs).mean(0).numpy()
    else:
        return np.zeros(glove.dim)

for idx, row in movies.iterrows():
    movie_id = row['movie_id']
    title = row['movie_title']
    features[movie_id] = text_to_vec(title)

np.save('movie_textual_features.npy', features)


# In[10]:


def extract_movie_textual_features(self):
    df = pd.read_csv(self.config.item_path)
    movie_ids = df['movie_id'].astype(int).tolist()
    glove = GloVe(name='6B', dim=300)
    features = np.zeros((max(movie_ids) + 1, 300))

    def text_to_vec(text):
        tokens = str(text).lower().replace('(', '').replace(')', '').replace(':', '').split()
        vecs = [glove[t] for t in tokens if t in glove.stoi]
        if vecs:
            return torch.stack(vecs).mean(0).numpy()
        else:
            return np.zeros(glove.dim)

    for idx, row in df.iterrows():
        movie_id = int(row['movie_id'])
        title = row['movie_title']
        features[movie_id] = text_to_vec(title)
    return features


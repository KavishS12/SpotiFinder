#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import skfuzzy as fuzz
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# In[2]:


df = pd.read_csv('./data/data.csv')
df.head()


# In[3]:


new_column_order = ['id', 'name', 'artists', 'year','popularity'] + [col for col in df.columns if col not in ['id', 'name', 'artists', 'year','popularity']]
df = df[new_column_order]
df


# In[4]:


df.shape


# In[5]:


df = df.drop(['duration_ms', 'release_date','liveness'], axis=1)
df.head()


# In[6]:


df[df.duplicated()].size


# In[7]:


df.info()


# In[8]:


print(f"Possible values for mode : {df['mode'].unique()}")
print(f"Possible values for key : {df['key'].unique()}")
print(f"Possible values for explicit : {df['explicit'].unique()}")
attributes = ['year','popularity']

for attr in attributes:
    attr_range = f"{df[attr].min()} – {df[attr].max()}"
    print(f"Range for {attr}: {attr_range}")


# In[9]:


X = df[['acousticness', 'danceability', 'energy', 'instrumentalness', 'loudness', 'speechiness', 'tempo', 'valence']].values


# In[10]:


n_clusters = 6
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    X.T, n_clusters, 2, error=0.005, maxiter=1000, init=None
)


# In[11]:


pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

for i in range(n_clusters):
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=u[i, :], label=f'Cluster {i}', alpha=0.5)

plt.scatter(pca.transform(cntr)[:, 0], pca.transform(cntr)[:, 1], marker='X', s=200, c='black', label='Centroids')
plt.legend()
plt.title("Fuzzy C-Means Clustering")
plt.show()


# In[12]:


def assign_clusters(u, threshold=0.35):
    clusters = []
    for i in range(u.shape[1]):
        clusters_for_point = [cluster for cluster in range(u.shape[0]) if u[cluster, i] >= threshold]
        clusters.append(clusters_for_point)
    return clusters

df['clusters'] = assign_clusters(u)


# In[13]:


df


# In[14]:


df['clusters'].value_counts()


# In[15]:


exploded_df = df.explode('clusters')

cluster_df = exploded_df.groupby("clusters").agg("mean")
cluster_df["count"] = exploded_df.groupby("clusters").size()
cluster_df = cluster_df[["count", "acousticness", "danceability", "energy", "instrumentalness", "loudness", "speechiness", "tempo", "valence"]]
cluster_df


# In[33]:


cluster_df.to_csv('./data/clusters_fuzzy.csv')


# # Clusters Analysis
# 
# ## Cluster 0: **Pop-Rock**
# * **Key Features**: High energy (0.57), fast tempo (147 BPM), electric-leaning (0.39 acousticness)
# * **Fit**: Energy and tempo match rock dynamics, while moderate danceability (0.53) reflects pop influence. Electric-heavy sound signature aligns with genre's core characteristics.
# 
# ## Cluster 1: **Indie Pop**
# * **Key Features**: High danceability (0.60), balanced acousticness (0.50), upbeat mood (0.57 valence) 
# * **Fit**: Even mix of acoustic/electric elements and moderate energy (0.49) captures indie pop's balanced, accessible sound. Medium tempo (111 BPM) suits the relaxed style.
# 
# ## Cluster 2: **Acoustic Folk / Classical**
# * **Key Features**: Highest acousticness (0.70), lowest energy (0.31), slowest tempo (76 BPM)
# * **Fit**: Strongly acoustic nature with high instrumentalness (0.26) reflects traditional instrumentation. Low energy and tempo match the contemplative style.
# 
# ## Cluster 3: **Alternative R&B**
# * **Key Features**: Balanced acousticness (0.50), moderate danceability (0.55), medium tempo (94 BPM)
# * **Fit**: Even distribution of features captures Alt-R&B's fusion style. Moderate energy (0.49) and valence (0.52) suit the genre's smooth, contemporary vibe.
# 
# ## Cluster 4: **EDM**
# * **Key Features**: Highest tempo (177 BPM), high energy (0.55), electronic-based (0.43 acousticness)
# * **Fit**: Fast pace and electronic emphasis perfectly match EDM characteristics. High valence (0.57) reflects the genre's energetic nature.
# 
# ## Cluster 5: **Dance Pop**
# * **Key Features**: High danceability (0.59), moderate energy (0.52), dance-optimized tempo (128 BPM)
# * **Fit**: Perfect balance of danceability and energy with mainstream tempo represents classic dance pop production style.
# 
# Each cluster shows distinct attribute patterns that naturally align with its assigned genre, creating clear separations in the musical space.

# In[17]:


features = ["acousticness", "danceability", "energy", "instrumentalness", "speechiness", "valence"]
plot_data = cluster_df[features]

plt.figure(figsize=(12, 6))
x = np.arange(0, len(features) * 2, 2)
width = 0.2

for i in range(len(plot_data)):
    values = plot_data.iloc[i]
    plt.bar(x + i * width, values, width, label=f'Cluster {i}')

plt.xlabel('Features')
plt.ylabel('Value')
plt.title('Music Cluster Feature Comparison')
plt.xticks(x + width * (len(plot_data) - 1) / 2, features, rotation=45)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.ylim(0, 0.7)
plt.tight_layout()
plt.show()


# ### Fuzzy C-Means (FCM) Evaluation for 6 Clusters
# 
# #### 1. **Fuzzy Partition Coefficient (FPC)**
# - **Range**: The FPC score ranges from 1/6(approx 0.167) (worst) to 1 (best).
# - **Interpretation**: A higher FPC score indicates well-defined clusters with less overlap.
#   
# #### 2. **Partition Entropy (PE)**
# - **Range**: The PE score ranges from 0 (best) to log(6)(approx 1.79) (worst).
# - **Interpretation**: Lower values of PE indicate that the membership of each data point to a single cluster is more certain, meaning that the clusters are more distinct.

# In[18]:


def fuzzy_partition_coefficient(U):
    C, N = U.shape
    FPC = np.sum(U**2) / N
    return FPC

def partition_entropy(U):
    C, N = U.shape
    PE = -np.sum(U * np.log(U)) / N
    return PE

fpc = fuzzy_partition_coefficient(u)
pe = partition_entropy(u)

print("Fuzzy Partition Coefficient (FPC):", fpc)
print("Partition Entropy (PE):", pe)


# In[19]:


exploded_df


# In[20]:


cluster_to_genre = {
    0: 'Pop-Rock',
    1: 'Indie Pop',
    2: 'Acoustic Folk/Classical',
    3: 'Alternative R&B',
    4: 'Electronic-Dance Music',
    5: 'Dance Pop'
}
genre_df = exploded_df.copy()
genre_df['genre'] = genre_df['clusters'].map(cluster_to_genre)
genre_df = genre_df.drop(columns='clusters')
genre_df.head()


# In[21]:


genre_df['genre'].isna().sum()


# In[22]:


genre_df = genre_df.dropna()


# In[23]:


from sklearn.preprocessing import MinMaxScaler

audio_feats = ["acousticness", "danceability", "energy", "instrumentalness", "loudness", "tempo","valence","speechiness"]
scaler = MinMaxScaler()
genre_df[audio_feats] = scaler.fit_transform(genre_df[audio_feats])


# In[24]:


genre_df


# In[25]:


genre_df.to_csv('./data/fuzzy_genre_df.csv')


# In[26]:


for i in audio_feats:
    print(i,genre_df[i].median())


# ### Filtering the rows which have max cluster membership less than a threshold _(not used later)_

# In[27]:


filtered_indices = [i for i in range(X.shape[0]) if np.max(u[:, i]) >= 0.70]
filtered_df = df.iloc[filtered_indices]
filtered_df.shape


# In[28]:


for i in filtered_indices:
    memberships = u[:, i]
    cluster = np.argmax(memberships)
    print(f"Data point {i} is most likely in cluster {cluster} with membership {memberships[cluster]}")


# In[29]:


filtered_df['cluster'] = [np.argmax(u[:, i]) for i in filtered_indices]
filtered_df.head()


# In[30]:


filtered_df['cluster'].value_counts().sort_index()


# In[31]:


features = ["acousticness", "danceability", "energy", "instrumentalness", "speechiness", "valence"]
plot_data = cluster_df[features]

plt.figure(figsize=(12, 6))
x = np.arange(0, len(features) * 1.5, 1.5)
width = 0.2

for i in range(len(plot_data)):
    values = plot_data.iloc[i]
    plt.bar(x + i * width, values, width, label=f'Cluster {i}')

plt.xlabel('Features')
plt.ylabel('Value')
plt.title('Music Cluster Feature Comparison')
plt.xticks(x + width * (len(plot_data) - 1) / 2, features, rotation=45)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.ylim(0, 0.7)
plt.tight_layout()
plt.show()


# In[32]:


from sklearn.metrics import silhouette_score
X = filtered_df[['acousticness', 'danceability', 'energy', 'instrumentalness', 'loudness', 'speechiness', 'tempo', 'valence']].values
clusters = filtered_df['cluster'].values
score = silhouette_score(X, clusters)
print(score)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# # Minor Project : Spotify Song Segmentation

# ## Reading the Dataset

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


sns.set_style("darkgrid")


# In[3]:


df = pd.read_csv("Spotify.csv")
df.head()


# In[4]:


df.rename(columns = {"track_popularity" : "popularity", "track_album_id" : "album_id", "track_album_name" : "album_name",
                    "track_album_release_date" : "release_date", "playlist_genre" : "genre", "playlist_subgenre" : "subgenre"},
                     inplace = True)
df.head()


# In[5]:


# Checking for null values

df.isnull().sum()


# In[6]:


df.dropna()


# --------------------------------------------------------------------------------------------------------------------------------

# ## Data Analysis

# **Top 5 most popular artists**

# In[7]:


top5 = df.groupby("track_artist").count().sort_values(by = "track_name", ascending = False)["track_name"][:5]
top5


# **All possible plots**

# In[8]:


plt.figure(figsize=(20,18), dpi= 80)
sns.pairplot(df, kind="scatter", plot_kws=dict(s=80, edgecolor="white", linewidth=2.5))
plt.show()


# **Correlation Matrix**

# In[9]:


cor = df.corr()
cor


# --------------------------------------------------------------------------------------------------------------------------------

# ## Data Preparation

# In[10]:


genre_data = df.copy()
genre_data.drop(columns = {"track_id", "track_name", "album_id", "album_name", "release_date", "playlist_name", "playlist_id"}, inplace = True)
genre_data.rename(columns = {"track_artist" : "artists"}, inplace = True)
genre_data


# The above data includes all the necessary features along with the artist name.

# In[11]:


df.drop_duplicates('track_name', inplace = True)
df.dropna(axis = 1)
df.dropna(axis = 0)


# In[12]:


# Performing Feature Scaling

feature_cols=['acousticness', 'danceability', 'duration_ms', 'energy',
              'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
              'speechiness', 'tempo', 'valence', 'loudness']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
normalized_df =scaler.fit_transform(df[feature_cols])

print(normalized_df[:2])


# --------------------------------------------------------------------------------------------------------------------------------

# ## Recommendation System using Cosine Similarity

# In[13]:


from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing

# Create a pandas series with song titles as indices and indices as series values
indices = pd.Series(df.index, index=df['track_name']).drop_duplicates()

# Create cosine similarity matrix based on given matrix
cosine = cosine_similarity(normalized_df)

def generate_recommendation(track_name, model_type=cosine ):
    """
    Purpose: Function for song recommendations
    Inputs: song title and type of similarity model
    Output: Pandas series of recommended songs
    """
    try:
        # Get song index for the input track_name
        index = indices[track_name]
    except KeyError:
        return "Song not found in the dataset"

    # Get list of songs for given songs
    score=list(enumerate(model_type[index]))

    # Sort the most similar songs
    similarity_score = sorted(score,key = lambda x:x[1],reverse = True)

    # Select the top-10 recommend songs
    similarity_score = similarity_score[1:11]
    recommended_indices = [i[0] for i in similarity_score]

    # Top 10 recommended songs
    recommended_songs =df['track_name'].iloc[recommended_indices]
    return recommended_songs


# In[27]:

# # without using streamlit
# input_track = input("Enter a song name : ")
# print("Recommended Songs:")
# print(generate_recommendation(input_track, cosine).values)


# In[28]:


import streamlit as st

# Define your recommendation function here (e.g., generate_recommendation)
# You'll need to import the necessary libraries and functions for recommendation.

# Input field for song name
input_track = st.text_input("Enter a song name:")

# Check if the user has entered a song name
if input_track:
    # if we want a button Generate and display recommendations when the user clicks a button
    # if st.button("Generate Recommendations"):

    # Call your recommendation function here
    recommendations = generate_recommendation(input_track, cosine)

    # Display recommendations
    st.header("Recommended Songs:")
    st.write(recommendations.values)


# In[ ]:





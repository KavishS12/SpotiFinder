# SpotiFinder - A Personalized Music Recommendation System 

SpotiFinder is a machine learning-powered music recommendation app that offers personalized music suggestions based on user-defined preferences. By combining content-based filtering with fuzzy clustering techniques, SpotiFinder delivers accurate music recommendations from Spotify's track dataset, providing users with control over their recommendations through adjustable filters and weights for each audio feature.

1. **Dataset**: Spotify dataset from Kaggle, containing audio features and metadata like track ID, artist, album, year, tempo, energy, acousticness,etc.

2. **Fuzzy C-Means Clustering**: Classifies songs into overlapping genre clusters, enhancing recommendation quality and allowing multi-genre recommendations.

3. **Content-Based Filtering**: Matches user-selected attributes (e.g., danceability, loudness) with songs in the dataset using a Nearest Neighbors model.

4. **User Preferences**: Adjustable weights for each feature and filters for genre, year, and popularity to refine recommendations.

5. **Recommendation Generation**: Based on Nearest Neighbors, songs are selected that best match user-defined criteria.

##
**Web App:** [SpotiFinder](https://spotifinder.streamlit.app/) 

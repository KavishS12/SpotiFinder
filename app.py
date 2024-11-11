import streamlit as st
import pandas as pd
import random
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
import streamlit.components.v1 as components

st.set_page_config(page_title="SpotiFinder", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("./data/fuzzy_genre_df.csv")
    return df

audio_feats = ["acousticness", "danceability", "instrumentalness", "loudness", "tempo", "speechiness"]
df = load_data()

def recommend_songs(start_year, end_year, test_feat, weights, selected_genres, min_popularity):
    # Filter by year and genre
    year_data = df[(df["year"] >= start_year) & (df["year"] <= end_year)]
    year_data = year_data[year_data["genre"].isin(selected_genres)]
    
    # Filter by popularity
    year_data = year_data[year_data["popularity"] >= min_popularity]
    year_data = year_data.drop_duplicates(subset='id')
    
    if year_data.empty:
        st.error("No songs found for the specified genre, popularity, and year range. Please adjust your selections.")
        return [], []
    
    # Apply weights to features
    weighted_features = (year_data[audio_feats] * weights).to_numpy()
    weighted_test_feat = [a * w for a, w in zip(test_feat, weights)]
    
    # Fit NearestNeighbors model
    model = NearestNeighbors()
    model.fit(weighted_features)
    n_neighbors = model.kneighbors([weighted_test_feat], n_neighbors=min(len(year_data),10), return_distance=False)[0]
    
    random_indices = list(n_neighbors)
    random.shuffle(random_indices)
    selected_neighbors = random_indices[:6]
    
    uris = year_data.iloc[selected_neighbors]["id"].tolist()
    audios = year_data.iloc[selected_neighbors][audio_feats].to_numpy()
    return uris, audios

def page():
    global df
    st.markdown("<h1 style='display:inline'>SpotiFinder</h1> <h3 style='display:inline; color:gray;'>Discover Your Perfect Playlist, One Song at a Time!</h3>", unsafe_allow_html=True)
    st.markdown('##')

    col1, col2, col3 = st.columns([10, 1, 10]) 
    
    with col1:
        st.markdown("### Select the Year Range:")
        start_year, end_year = st.slider(
            'Select the year range',
            1921, 2020, (1980, 2020)
        )
        
    with col3:
        st.markdown("### Select Minimum Popularity:")
        min_popularity = st.slider('Minimum Popularity', 0, 100, 50)
    
    # Left sidebar for audio feature sliders
    with st.sidebar:
        st.markdown("# Select Features to Customize:")
        acousticness = st.slider('Acousticness', 0.0, 1.0, 0.5)
        danceability = st.slider('Danceability', 0.0, 1.0, 0.5)
        instrumentalness = st.slider('Instrumentalness', 0.0, 1.0, 0.2)
        loudness = st.slider('Loudness (in dB)', -60.0, 4.0, -20.0)
        tempo = st.slider('Tempo(in bpm)', 0.0, 244.0, 120.0)
        speechiness = st.slider('Speechiness', 0.0, 1.0, 0.5)

        st.markdown('##')
        with st.expander("Adjust Feature Weights", expanded=False):
            acousticness_weight = st.slider('Weight for Acousticness', 0.0, 2.0, 1.0)
            danceability_weight = st.slider('Weight for Danceability', 0.0, 2.0, 1.0)
            instrumentalness_weight = st.slider('Weight for Instrumentalness', 0.0, 2.0, 1.0)
            loudness_weight = st.slider('Weight for Loudness', 0.0, 2.0, 1.0)
            tempo_weight = st.slider('Weight for Tempo', 0.0, 2.0, 1.0)
            speechiness_weight = st.slider('Weight for Speechiness', 0.0, 2.0, 1.0)
    
    # Genre selection outside sidebar (main content area)
    st.markdown("### Select Genre(s)")
    genres = df["genre"].unique().tolist()
    selected_genres = st.multiselect(" Choose one or more genres", options=genres, default=genres)
    st.markdown("#")

    test_feat = [acousticness, danceability, instrumentalness, loudness, tempo, speechiness]
    weights = [acousticness_weight, danceability_weight, instrumentalness_weight, loudness_weight, tempo_weight, speechiness_weight]
    
    uris, audios = recommend_songs(start_year, end_year, test_feat, weights, selected_genres, min_popularity)

    if uris:
        st.write("### Recommended Songs:")
        tracks_per_page = 6
        tracks = [
            f'<iframe src="https://open.spotify.com/embed/track/{uri}" width="260" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>'
            for uri in uris
        ]

        if 'start_track_i' not in st.session_state:
            st.session_state['start_track_i'] = 0
        
        with st.container():
            col1, col2, col3 = st.columns([2,1,2])

            current_tracks = tracks[st.session_state['start_track_i']: st.session_state['start_track_i'] + tracks_per_page]
            current_audios = audios[st.session_state['start_track_i']: st.session_state['start_track_i'] + tracks_per_page]
            if st.session_state['start_track_i'] < len(tracks):
                for i, (track, audio) in enumerate(zip(current_tracks, current_audios)):
                    if i % 2 == 0:
                        with col1:
                            components.html(track, height=400)
                            with st.expander("See more details"):
                                audio_df = pd.DataFrame(dict(r=audio[:6], theta=audio_feats[:6]))
                                fig = px.line_polar(audio_df, r='r', theta='theta', line_close=True)
                                fig.update_layout(height=400, width=340)
                                st.plotly_chart(fig)
                    else:
                        with col3:
                            components.html(track, height=400)
                            with st.expander("See more details"):
                                audio_df = pd.DataFrame(dict(r=audio[:6], theta=audio_feats[:6]))
                                fig = px.line_polar(audio_df, r='r', theta='theta', line_close=True)
                                fig.update_layout(height=400, width=340)
                                st.plotly_chart(fig)
            else:
                st.write("No songs left to recommend")

page()

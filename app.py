import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
import streamlit.components.v1 as components

st.set_page_config(page_title="SpotifyFinder", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("data_with_genres.csv")
    return df

audio_feats = ["acousticness", "danceability", "energy", "instrumentalness", "loudness", "speechiness", "tempo", "valence"]

df = load_data()

def recommend_songs(start_year, end_year, test_feat):
    # Filter the dataset by the release year range
    year_data = df[(df["year"] >= start_year) & (df["year"] <= end_year)]
    if year_data.empty:
        st.error("No songs found in the specified year range. Please adjust the year range.")
        return [], []
    
    # Fit Nearest Neighbors on the filtered dataset
    model = NearestNeighbors()
    model.fit(year_data[audio_feats])
    n_neighbors = model.kneighbors([test_feat], n_neighbors=len(year_data), return_distance=False)[0]
    
    # Get URIs and audio features of recommended songs
    uris = year_data.iloc[n_neighbors]["id"].tolist()
    audios = year_data.iloc[n_neighbors][audio_feats].to_numpy()
    return uris, audios

def page():
    st.title("SpotiFinder")
    st.write("Customize your listening experience based on several key audio features. Adjust the settings to discover new music tailored to your preferences!")
    st.markdown("##")

    with st.sidebar:
        st.markdown("### Select Features to Customize:")
        start_year, end_year = st.slider(
            'Select the year range',
            1920, 2021, (2000, 2020)
        )
        acousticness = st.slider('Acousticness', 0.0, 1.0, 0.5)
        danceability = st.slider('Danceability', 0.0, 1.0, 0.5)
        energy = st.slider('Energy', 0.0, 1.0, 0.5)
        instrumentalness = st.slider('Instrumentalness', 0.0, 1.0, 0.0)
        liveness = st.slider('Liveness', 0.0, 1.0, 0.3)
        loudness = st.slider('Loudness', -60.0, 0.0, -20.0)
        speechiness = st.slider('Speechiness', 0.0, 1.0, 0.1)
        tempo = st.slider('Tempo', 0.0, 244.0, 120.0)
        valence = st.slider('Valence', 0.0, 1.0, 0.5)

    test_feat = [acousticness, danceability, energy, instrumentalness, liveness, loudness, speechiness, tempo, valence]
    uris, audios = recommend_songs(start_year, end_year, test_feat)

    if uris:
        st.write("### Recommended Songs:")
        tracks_per_page = 6
        tracks = [
            f'<iframe src="https://open.spotify.com/embed/track/{uri}" width="260" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>'
            for uri in uris
        ]

        current_tracks = tracks[:tracks_per_page]
        current_audios = audios[:tracks_per_page]

        for i, (track, audio) in enumerate(zip(current_tracks, current_audios)):
            with st.container():
                if i % 2 == 0:
                    components.html(track, height=400)
                    with st.expander("See more details"):
                        df = pd.DataFrame(dict(r=audio[:5], theta=audio_feats[:5]))
                        fig = px.line_polar(df, r='r', theta='theta', line_close=True)
                        fig.update_layout(height=400, width=340)
                        st.plotly_chart(fig)

page()

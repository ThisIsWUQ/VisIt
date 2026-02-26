# Fix the issue: "unsupported version of sqlite3. Chroma requires sqlite3 >= 3.35.0."
# reference: https://discuss.streamlit.io/t/issues-with-chroma-and-sqlite/47950/4
# ---------------------------------------------------------------------------------
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# ---------------------------------------------------------------------------------

import streamlit as st
import numpy as np
import requests
import chromadb
import torch
from chromadb.utils import embedding_functions
from TMDB import api_response
from YTS_url import get_movie_page_url
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from transformers import BlipProcessor, BlipForConditionalGeneration
#from transformers import pipeline

# Initialize an image-to-text model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Initialize ChromaDB client and collection
# reference: https://github.com/TharinduMadhusanka/semantic-movie-search/blob/main/app.py
# ---------------------------------------------------------------------------------
chroma_client = chromadb.PersistentClient(path="tmdbtopmovies")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L12-v2")
collection = chroma_client.get_or_create_collection(name="movies_collection", embedding_function=sentence_transformer_ef)
# ---------------------------------------------------------------------------------

# Set up app's name
st.set_page_config(
    page_title="DoodleFindFilm",
    page_icon=":film_frames:"
)

# App title and brief description
st.title("DoodleFindFilm")
st.write("Welcome to DoodleFindFilm! Find a Film :film_frames: from your Scribbles :pencil2:")

# Instruction
st.header("Instruction", divider=True)
st.write("1. Upload your scribbles or draw pictures of object relating to the film you are looking for in the canvas below")
st.write("2. Optional, filter your results by selecting minimum release year in the slide bars below")
st.write("3. Click the button 'Search' to find the film")

# Adjust searching
# reference: https://github.com/TharinduMadhusanka/semantic-movie-search/blob/main/app.py
# ---------------------------------------------------------------------------------
col3, col4 = st.columns([1, 1])
with col3:
    n_movies = st.slider("Select number of movies to display:", 1, 20,3)
with col4:
    min_year = st.slider("Select minimum release year:", 1900, 2025, 1900)
# ---------------------------------------------------------------------------------

# Drawing canvas
# reference: https://github.com/andfanilo/streamlit-drawable-canvas
# ---------------------------------------------------------------------------------
# Specify canvas parameters in application
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "point", "line", "rect", "circle", "transform")
)

stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
if drawing_mode == 'point':
    point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
st.sidebar.write("Note: After choosing the stroke's color, close the colorpicker widget by clicking anywhere on the screen")
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])

realtime_update = st.sidebar.checkbox("Update in realtime", True)

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_image=Image.open(bg_image) if bg_image else None,
    update_streamlit=realtime_update,
    height=500,
    width=1000,
    drawing_mode=drawing_mode,
    point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
    key="canvas",
)
# ---------------------------------------------------------------------------------

# Generate a caption for the given image
# reference: ChatGPT https://chatgpt.com/share/678fc3f3-b94c-800b-b114-407634477991
# ---------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def generate_caption(image):
    resized_image = image.resize((256, 256))  # Resize for efficiency
    # Make captions from the picture drawn
    # reference: https://huggingface.co/tasks/image-to-text
    # ---------------------------------------------------------------------------------
    #captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    # ---------------------------------------------------------------------------------
    #return captioner(resized_image)[0]['generated_text']
    inputs = processor(images=resized_image, return_tensors="pt")
    out = model.generate(**inputs)
    
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption
# ---------------------------------------------------------------------------------

# Search for Results
# reference: https://github.com/TharinduMadhusanka/semantic-movie-search/blob/main/app.py
# ---------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def search_movie(query, n_movies, min_year):
    return collection.query(query_texts=[query], n_results=n_movies, where={"release_year": {"$gte": min_year}})

if st.button("Search"):
    # Caption the scribbles drawn
    if canvas_result.image_data is not None:
        # Convert the numpy array to an image
        # reference: ChatGPT https://chatgpt.com/share/678f9b36-c14c-800b-a3e3-e7a768d4c32f
        # ---------------------------------------------------------------------------------
        image = Image.fromarray((canvas_result.image_data).astype(np.uint8))
        # ---------------------------------------------------------------------------------
        query = generate_caption(image)  # Cached function call
        st.write(query)
    
    results = search_movie(query, n_movies, min_year)

    for i, result in enumerate(results["metadatas"][0]):
        movie_id = results['ids'][0][i]
        api_data = api_response(movie_id)
        if not api_data:
            st.write("Error fetching data for movie:", result['title'])
            continue
        imdb_url = api_data["imdb_url"] if "imdb_url" in api_data else None
        tmdb_url = f"https://www.themoviedb.org/movie/{movie_id}"
        yts_url = get_movie_page_url(result['title']) 

        st.markdown(f"### [{result['title']}]({tmdb_url})")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(api_data['poster_url'], width=300)
        with col2:
            st.markdown(f"**Genres:** {(result['genres'])}")
            st.markdown(f"**Release Year:** {result['release_year']}")
            st.markdown(f"**Runtime:** {result['runtime']} minutes")
            st.markdown(f"**Overview:** {results['documents'][0][i]}")
            if imdb_url:
                st.markdown(f"**[IMDB]({imdb_url})**")
            if yts_url:
                st.markdown(f"**[YTS]({yts_url})**")
        st.markdown("---")
# ---------------------------------------------------------------------------------    
    # Take a Survey
    st.header("Take a Survey", divider=True)
    st.write("How is your impression with the app? If you have 5 minutes, please take this survey below")
    st.write("Also, do not close this app yet! You can close it after taking the survey.")
    st.link_button("Click here to go to the survey", "https://forms.gle/Cbya8epun8ngyeX4A")





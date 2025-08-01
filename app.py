from dotenv import load_dotenv
import streamlit as st
import re
import requests
from bs4 import BeautifulSoup
from wordcloud import WordCloud, STOPWORDS
import textstat
import matplotlib.pyplot as plt

from core import (
    get_language_name,
    import_google_api,
    embedding_function,
    persistent_client,
    get_article,
    get_article_hr
)

# Streamlit App Configuration - This must be first
st.set_page_config(page_title="Polleo FAQ", page_icon="", layout="wide")

#import fasttext
# Load only once
#@st.cache_resource
#def load_fasttext_model():
#   return fasttext.load_model("models/lid.176.bin")
#ft_lang_model = load_fasttext_model()

from lingua import Language, LanguageDetectorBuilder
@st.cache_resource
def load_language_detector():
    detector = LanguageDetectorBuilder.from_languages(
    Language.CROATIAN,
    Language.ENGLISH,
    ).build()
    return detector

def detect_language_lingua(text):
    detector = load_language_detector()
    lang = detector.detect_language_of(text)
    return lang.name if lang else "Unknown"


#from lingua import Language, LanguageDetectorBuilder
#@st.cache_resource
#def load_language_detector():
    #return LanguageDetectorBuilder.from_all_languages().build()

# Function to detect language
#def detect_language_lingua(text):
    #detector = load_language_detector()
    #lang = detector.detect_language_of(text)
    #return lang.name if lang else "Unknown"
    #print(lang.name)

#user_query = "Objasni logiku u Abelarda"
#user_language = detect_language_lingua(user_query)
#st.write(f"Text: {user_query}")
#st.write(f"Detected Language: {user_language}")

# -----------------------------------
# Importing Google API key, embedding function, collection, and retry
# -----------------------------------
client = import_google_api()
gemini_embedding_function = embedding_function(client)
embed_fn, collection = persistent_client(gemini_embedding_function)

# -----------------------------------
# Streamlit UI
# -----------------------------------


def main():

    st.title("Polleo FAQ")  # 👈 title next to logo

    st.markdown("Pretraživanje FAQ baze Polleo Sporta")

    with st.form(key="query_form"):
        user_query = st.text_input("💬 Pitajte FAQ:", placeholder="e.g., Kako mogu naručiti?")
        submit_button = st.form_submit_button("🔎 Pretražite FAQ")
        user_language = "HR"

    if submit_button and user_query:
        with st.spinner("Searching for the article..."):
            article = get_article_hr(user_query, embed_fn, collection, client, user_language)

        # Save into session_state
        st.session_state['article'] = article
        st.session_state['user_query'] = user_query

    # Always display article if it exists in session state
    if 'article' in st.session_state:
        st.subheader("📖 Odgovor")
        st.markdown(st.session_state['article'])


if __name__ == "__main__":
    main()

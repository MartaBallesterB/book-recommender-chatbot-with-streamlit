import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from src.main import load_books_dataset, build_tfidf, build_embeddings, build_chroma, build_generator, recommend_tfidf, recommend_embeddings, recommend_chroma, generate_response

from dotenv import load_dotenv
load_dotenv() # loads HuggingFace token from .env
HF_TOKEN = os.environ.get("HF_TOKEN", "")

@st.cache_resource
def get_books():
    return load_books_dataset()

@st.cache_resource
def get_tfidf_setup():
    books = get_books()
    vectorizer, book_vectors = build_tfidf(books)
    return vectorizer, book_vectors

@st.cache_resource
def get_embeddings_setup():
    books = get_books()
    embedder, book_vectors = build_embeddings(books)
    return embedder, book_vectors

@st.cache_resource
def get_chroma_setup():
    books = get_books()
    return build_chroma(books, hf_token=HF_TOKEN)

@st.cache_resource
def get_generator():
    return build_generator(hf_token=HF_TOKEN)

st.sidebar.title("Recommender settings")
mode = st.sidebar.radio("Mode", ["ChromaDB + LLM", "ChromaDB", "Embeddings", "TF-IDF"])
top_N = st.sidebar.slider("Number of recommendations", min_value=1, max_value=10, value=5)

st.title("Welcome to my Book Recommender! 📚🤖")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Let me help you find your next great read. Just tell me what kind of story are you looking for."}
    ]

MAX_LLM_HISTORY_TURNS = 10

# only for chromadb + LLM mode
if "llm_messages" not in st.session_state:
    st.session_state.llm_messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if query := st.chat_input("What kind of story are you looking for?"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    if mode == "Embeddings":
        books = get_books()
        embedder, book_vectors = get_embeddings_setup()
        results = recommend_embeddings(query, top_N, books, embedder, book_vectors)
    elif mode in ("ChromaDB", "ChromaDB + LLM"):
        store = get_chroma_setup()
        results = recommend_chroma(query, top_N, store)
    else:
        books = get_books()
        vectorizer, book_vectors = get_tfidf_setup()
        results = recommend_tfidf(query, top_N, books, vectorizer, book_vectors)

    if results.empty:
        response = "I couldn't find any books matching your query. Try different keywords please!"
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.write(response)
    else:
        with st.chat_message("assistant"):
            if mode == "ChromaDB + LLM":
                generator = get_generator()
                llm_response = generate_response(query, results, generator, llm_history=st.session_state.llm_messages)

                st.session_state.llm_messages.append({"role": "user", "content": query})
                st.session_state.llm_messages.append({"role": "assistant", "content": llm_response})
                max_msgs = MAX_LLM_HISTORY_TURNS * 2
                if len(st.session_state.llm_messages) > max_msgs:
                    st.session_state.llm_messages = st.session_state.llm_messages[-max_msgs:]

                st.session_state.messages.append({"role": "assistant", "content": llm_response})
                st.write(llm_response)

            header = f"Here are my top {len(results)} recommendations *(mode: {mode})*:"
            st.session_state.messages.append({"role": "assistant", "content": header})
            st.write(header)
            for i, row in results.iterrows():
                title_line = f"**{i + 1}. {row['title']}** by {row['author']}"
                if row["genres"]:
                    title_line += f"  \n_{row['genres']}_"
                if "score" in results.columns:
                    title_line += f"  \nSimilarity score: {row['score']}"
                st.markdown(title_line)
                if row.get("summary"):
                    snippet = row["summary"][:500].rsplit(" ", 1)[0] + "…"
                    with st.expander("Summary"):
                        st.write(snippet)

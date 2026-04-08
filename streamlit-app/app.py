import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from src.main import top_N_book_recommender

st.title("Welcome to my Book Recommender! 📚🤖")

top_N = 5

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm here to help you find your next great read. Just tell me what kind of story are you looking for, and I'll recommend the best 3 options for you."}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if query := st.chat_input("What kind of story are you looking for?"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    results = top_N_book_recommender(query, top_N)

    response = f"Here are my top {top_N} recommendations:\n\n"
    for i, row in results.iterrows():
        response += f"**{i + 1}. {row['title']}** by {row['author']}\n"
        if row["genres"]:
            response += f"_{row['genres']}_\n"
        response += "\n"

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)

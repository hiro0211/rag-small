"""RAG Small - Streamlit Chat UI."""

import streamlit as st
from dotenv import load_dotenv

load_dotenv(".env.local")

from lib.chat import generate_response

st.set_page_config(
    page_title="RAG Small",
    page_icon="🔍",
    layout="centered",
)

st.title("RAG Small")
st.caption("社内ナレッジに基づいて質問に回答するアシスタント")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("質問を入力してください"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and stream response
    with st.chat_message("assistant"):
        response = st.write_stream(generate_response(prompt))

    st.session_state.messages.append({"role": "assistant", "content": response})

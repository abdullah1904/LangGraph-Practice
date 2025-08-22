import streamlit as st
from Backend import graph as chatbot, HumanMessage

CONFIG = {
    'configurable':{
        'thread_id': 1
    }
}

if 'message_history' not in st.session_state:
    st.session_state.message_history = []

for message in st.session_state.message_history:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.text(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.text(message["content"])

userInput = st.chat_input("Type your message here...")

if userInput:
    st.session_state.message_history.append({"role": "user", "content": userInput})
    with st.chat_message("user"):
        st.text(userInput)
    with st.chat_message("assistant"):
        response = st.write_stream(
            message_chunk.content for message_chunk, meta_data in chatbot.stream(
            { "messages": [HumanMessage(content=userInput)]},
            config=CONFIG,
            stream_mode='messages'
        ))
    st.session_state.message_history.append({"role": "assistant", "content": response})
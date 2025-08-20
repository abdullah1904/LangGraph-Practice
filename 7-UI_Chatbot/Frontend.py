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
    response = chatbot.invoke({'messages': HumanMessage(content=userInput)}, config=CONFIG)
    st.session_state.message_history.append({"role": "assistant", "content": response['messages'][-1].content})
    with st.chat_message("assistant"):
        st.markdown(response['messages'][-1].content)
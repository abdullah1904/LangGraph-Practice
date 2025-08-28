import streamlit as st
from Backend import graph as chatbot, HumanMessage
import uuid

def generate_thread_id():
    return uuid.uuid4()

def reset_chat():
    st.session_state.message_history = []
    thread_id = generate_thread_id()
    st.session_state.thread_id = thread_id
    add_thread(thread_id)

def add_thread(thread_id):
    if thread_id not in st.session_state.conversation_threads:
        st.session_state.conversation_threads.append(thread_id)

def load_conversation(thread_id): 
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get('messages', [])

if 'message_history' not in st.session_state:
    st.session_state.message_history = []

if 'thread_id' not in st.session_state:
    st.session_state.thread_id = generate_thread_id()

if 'conversation_threads' not in st.session_state:
    st.session_state.conversation_threads = []

add_thread(st.session_state.thread_id)

st.sidebar.title("Resume Chatbot")
if st.sidebar.button("New Conversation"):
    reset_chat()

st.sidebar.header("Conversations")

for thread_id in st.session_state.conversation_threads[::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state.thread_id = thread_id
        messages = load_conversation(thread_id)
        temp_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                temp_messages.append({"role": "user", "content": message.content})
            else:
                temp_messages.append({"role": "assistant", "content": message.content})
        st.session_state.message_history = temp_messages

for message in st.session_state.message_history:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.text(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

userInput = st.chat_input("Type your message here...")

if userInput:
    st.session_state.message_history.append({"role": "user", "content": userInput})
    with st.chat_message("user"):
        st.text(userInput)
    with st.chat_message("assistant"):
        response = st.write_stream(
            message_chunk.content for message_chunk, meta_data in chatbot.stream(
            { "messages": [HumanMessage(content=userInput)]},
            config= {
                'configurable':{
                    'thread_id': st.session_state.thread_id
                }
            },
            stream_mode='messages'
        ))
    st.session_state.message_history.append({"role": "assistant", "content": response})
import streamlit as st
from Backend import graph as chatbot, HumanMessage, AIMessage, ToolMessage,  listThreads
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
    st.session_state.conversation_threads = listThreads()

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
        status_holder = {"box": None}
        def ai_only_stream():
            for message_chunk, meta_data in chatbot.stream(
                { "messages": [HumanMessage(content=userInput)]},
                config= {
                    'configurable':{
                        'thread_id': st.session_state.thread_id,
                        'user_id': 12345678
                    },
                    'metadata': {
                        'thread_id': st.session_state.thread_id,
                        'user_id': 12345678
                    },
                    "run_name": "chatbot_graph_run"
                },
                stream_mode='messages'
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content
        ai_message = st.write_stream(ai_only_stream()) 
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )
    st.session_state.message_history.append({"role": "assistant", "content": ai_message})
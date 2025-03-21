import streamlit as st
from dataclasses import dataclass
from typing import Literal

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_openai import ChatOpenAI

# Message structure
@dataclass
class Message:
    origin: Literal["human", "ai"]
    message: str

# Load embedded CSS for chat UI
def load_css():
    st.markdown("""
    <style>
    .chat-row { display: flex; margin: 5px; width: 100%; }
    .row-reverse { flex-direction: row-reverse; }
    .chat-bubble {
        font-family: "Source Sans Pro", sans-serif;
        padding: 8px 12px;
        margin: 4px 10px;
        max-width: 70%;
    }
    .ai-bubble {
        background: #f1f4f8;
        color: #333;
        border-radius: 12px;
    }
    .human-bubble {
        background: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%);
        color: #003049;
        border-radius: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Session state
def init_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = "user-session"

# Memory store
store = {}
def get_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
        store[session_id].add_ai_message(
            "Hello, I’m MindEase 🧘 – your gentle relaxation companion. "
            "What’s been weighing on your mind?"
        )
    return store[session_id]

# LLM setup
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=st.secrets["openai_api_key"],
    temperature=0.6
)

# Prompt + chain
prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are MindEase, a calm and caring AI therapist. "
     "Help users reduce stress and anxiety. Use mindfulness, empathy, and gentle questions."),
    MessagesPlaceholder(variable_name="messages")
])

chain = prompt | llm

# Wrap with memory
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history=get_history,
    input_messages_key="messages",
    history_messages_key="messages"
)

# Handle submit
def on_click():
    user_input = st.session_state.human_prompt
    st.session_state.history.append(Message("human", user_input))

    response = chain_with_history.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config={"configurable": {"session_id": st.session_state.session_id}}
    )

    st.session_state.history.append(Message("ai", response.content))

# App start
init_state()
load_css()

st.title("🧘 MindEase: Your Relaxation Companion")
st.markdown("Let’s work through your stress — gently, mindfully, and one breath at a time.")

chat_placeholder = st.container()
prompt_placeholder = st.form("chat-form")

# Chat history display
with chat_placeholder:
    for chat in st.session_state.history:
        div = f"""
        <div class="chat-row {'row-reverse' if chat.origin == 'human' else ''}">
            <div class="chat-bubble {'human-bubble' if chat.origin == 'human' else 'ai-bubble'}">
                {chat.message}
            </div>
        </div>
        """
        st.markdown(div, unsafe_allow_html=True)

# Input form
with prompt_placeholder:
    st.markdown("**What’s on your mind?**")
    cols = st.columns((6, 1))
    cols[0].text_input("Your message", key="human_prompt", label_visibility="collapsed")
    cols[1].form_submit_button("Submit", type="primary", on_click=on_click)

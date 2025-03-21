import streamlit as st
from dataclasses import dataclass
from typing import Literal

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.memory import ChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain_community.callbacks.manager import get_openai_callback

# --- Basic chat formatting
@dataclass
class Message:
    origin: Literal["human", "ai"]
    message: str

# --- Memory store
store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
        store[session_id].add_ai_message(
            "Hello. I’m MindEase 🧘— your relaxation companion. "
            "Tell me what’s weighing on your mind today."
        )
    return store[session_id]

# --- Styles
def load_css():
    css = """
    <style>
    .chat-row { display: flex; margin: 5px; width: 100%; }
    .row-reverse { flex-direction: row-reverse; }
    .chat-bubble {
        font-family: "Source Sans Pro", sans-serif;
        border: 1px solid transparent;
        padding: 5px 10px;
        margin: 0px 7px;
        max-width: 70%;
    }
    .ai-bubble {
        background: #f7f9fc;
        color: #222;
        border-radius: 12px;
    }
    .human-bubble {
        background: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%);
        color: #003049;
        border-radius: 20px;
    }
    .chat-icon { border-radius: 5px; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# --- Init app state
def init():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0
    if "session_id" not in st.session_state:
        st.session_state.session_id = "user-session"

# --- LLM & Chain setup
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.6,
    openai_api_key=st.secrets["openai_api_key"]
)

prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are MindEase, a calm and compassionate AI therapist. "
     "Help users manage stress, guide mindfulness, and speak soothingly. "
     "Avoid giving medical advice. Gently ask questions and validate feelings."),
    MessagesPlaceholder(variable_name="messages")
])

chain = prompt | llm

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
    history_messages_key="messages"
)

# --- Callback
def on_click_callback():
    user_input = st.session_state.human_prompt
    st.session_state.history.append(Message("human", user_input))

    with get_openai_callback() as cb:
        response = chain_with_history.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config={"configurable": {"session_id": st.session_state.session_id}}
        )
        st.session_state.history.append(Message("ai", response.content))
        st.session_state.token_count += cb.total_tokens

# --- App
load_css()
init()

st.title("🧘 MindEase: Your Relaxation Companion")
st.markdown("Let’s work through your stress, one breath at a time.")

chat_placeholder = st.container()
prompt_placeholder = st.form("chat-form")
info = st.empty()

with chat_placeholder:
    for chat in st.session_state.history:
        div = f"""
        <div class="chat-row {'row-reverse' if chat.origin == 'human' else ''}">
            <img class="chat-icon" src="app/static/{'user_icon.png' if chat.origin == 'human' else 'ai_icon.png'}"
                 width=32 height=32>
            <div class="chat-bubble {'human-bubble' if chat.origin == 'human' else 'ai-bubble'}">
                &#8203;{chat.message}
            </div>
        </div>
        """
        st.markdown(div, unsafe_allow_html=True)

with prompt_placeholder:
    st.markdown("**What’s on your mind?**")
    cols = st.columns((6, 1))
    cols[0].text_input(
        "Your message",
        value="I'm feeling anxious today...",
        label_visibility="collapsed",
        key="human_prompt",
    )
    cols[1].form_submit_button("Submit", type="primary", on_click=on_click_callback)

info.caption(f"Used {st.session_state.token_count} tokens")

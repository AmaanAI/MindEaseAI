import streamlit as st
from dataclasses import dataclass
from typing import Literal
from langchain_core.chat_history import InMemoryChatMessageHistory

# from langchain_core.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
import streamlit.components.v1 as components

@dataclass
class Message:
    """Class for keeping track of a chat message."""
    origin: Literal["human", "ai"]
    message: str

def load_css():
    st.markdown(
        """
        <style>
        .chat-row {
            display: flex;
            margin: 5px;
            width: 100%;
        }
        .row-reverse {
            flex-direction: row-reverse;
        }
        .chat-bubble {
            font-family: "Source Sans Pro", sans-serif, "Segoe UI", "Roboto", sans-serif;
            border: 1px solid transparent;
            padding: 5px 10px;
            margin: 0px 7px;
            max-width: 70%;
        }
        .ai-bubble {
            background: rgb(240, 242, 246);
            border-radius: 10px;
        }
        .human-bubble {
            background: linear-gradient(135deg, rgb(0, 178, 255) 0%, rgb(0, 106, 255) 100%);
            color: white;
            border-radius: 20px;
        }
        .chat-icon {
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0
    if "conversation" not in st.session_state:
        llm = ChatOpenAI(
            temperature=0.6,
            openai_api_key=st.secrets["openai_api_key"],
            model="gpt-3.5-turbo"
        )
        memory = ConversationBufferMemory(
            chat_memory=InMemoryChatMessageHistory(),
            return_messages=True
        )
        st.session_state.conversation = ConversationChain(
            llm=llm,
            memory=memory,
        )

def on_click_callback():
    human_prompt = st.session_state.human_prompt
    llm_response = st.session_state.conversation.predict(input=human_prompt)
    st.session_state.history.append(Message("human", human_prompt))
    st.session_state.history.append(Message("ai", llm_response))
    # Token counting can be implemented based on your specific requirements

load_css()
initialize_session_state()

st.title("🧘 MindEase: Your Relaxation Companion")
st.markdown("Feeling overwhelmed or stressed? Let me help you relax, breathe, and find your calm.")

chat_placeholder = st.container()
prompt_placeholder = st.form("chat-form")
credit_card_placeholder = st.empty()

with chat_placeholder:
    for chat in st.session_state.history:
        div = f"""
<div class="chat-row {'row-reverse' if chat.origin == 'human' else ''}">
    <img class="chat-icon" src="{'user_icon.png' if chat.origin == 'human' else 'ai_icon.png'}" width=32 height=32>
    <div class="chat-bubble {'human-bubble' if chat.origin == 'human' else 'ai-bubble'}">
        {chat.message}
    </div>
</div>
        """
        st.markdown(div, unsafe_allow_html=True)
    for _ in range(3):
        st.markdown("")

with prompt_placeholder:
    st.markdown("**Chat**")
    cols = st.columns((6, 1))
    cols[0].text_input(
        "Chat",
        value="Hello bot",
        label_visibility="collapsed",
        key="human_prompt",
    )
    cols[1].form_submit_button(
        "Submit",
        type="primary",
        on_click=on_click_callback,
    )

# Token usage display can be implemented based on your specific requirements

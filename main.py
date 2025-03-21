from dataclasses import dataclass
from typing import Literal
import streamlit as st
from langchain import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryMemory
import streamlit.components.v1 as components

@dataclass
class Message:
    """Class for keeping track of a chat message."""
    origin: Literal["human", "ai"]
    message: str

def load_css():
    css = """
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
        background: #f7f9fc;
        color: #222;
        border-radius: 12px;
    }

    .human-bubble {
        background: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%);
        color: #003049;
        border-radius: 20px;
    }

    .chat-icon {
        border-radius: 5px;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0
    if "conversation" not in st.session_state:
        llm = OpenAI(
            temperature=0.6,
            openai_api_key=st.secrets["openai_api_key"],
            model_name="gpt-3.5-turbo"
        )
        st.session_state.conversation = ConversationChain(
            llm=llm,
            memory=ConversationSummaryMemory(llm=llm),
            verbose=False
        )
        intro_context = (
            "You are MindEase, a calm, gentle, and compassionate AI therapist. "
            "Your goal is to help the user feel relaxed, less anxious, and more centered. "
            "Speak in a soothing tone. Use mindfulness techniques like breathing exercises, visualizations, and affirmations. "
            "Do not diagnose or give medical advice. Be warm, encouraging, and present."
        )
        st.session_state.conversation.memory.buffer = intro_context

def on_click_callback():
    with get_openai_callback() as cb:
        human_prompt = st.session_state.human_prompt
        llm_response = st.session_state.conversation.run(human_prompt)
        st.session_state.history.append(Message("human", human_prompt))
        st.session_state.history.append(Message("ai", llm_response))
        st.session_state.token_count += cb.total_tokens

# --- App Layout ---
load_css()
initialize_session_state()

st.title("🧘 MindEase: Your Relaxation Companion")
st.markdown("Feeling overwhelmed or stressed? Let me help you relax, breathe, and find clarity.")

chat_placeholder = st.container()
prompt_placeholder = st.form("chat-form")
credit_card_placeholder = st.empty()

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

    for _ in range(3):
        st.markdown("")

with prompt_placeholder:
    st.markdown("**What's on your mind?**")
    cols = st.columns((6, 1))
    cols[0].text_input(
        "Chat",
        value="I’m feeling a bit overwhelmed today...",
        label_visibility="collapsed",
        key="human_prompt",
    )
    cols[1].form_submit_button(
        "Submit",
        type="primary",
        on_click=on_click_callback,
    )

credit_card_placeholder.caption(f"""
Used {st.session_state.token_count} tokens  
Debug Langchain conversation memory:  
{st.session_state.conversation.memory.buffer}
""")

# Optional: Add Enter-to-submit enhancement
components.html("""
<script>
const streamlitDoc = window.parent.document;
const buttons = Array.from(
    streamlitDoc.querySelectorAll('.stButton > button')
);
const submitButton = buttons.find(el => el.innerText === 'Submit');
streamlitDoc.addEventListener('keydown', function(e) {
    if (e.key === 'Enter') {
        submitButton.click();
    }
});
</script>
""", height=0, width=0)

import streamlit as st
from dataclasses import dataclass
from typing import Literal

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Chat message structure ---
@dataclass
class Message:
    origin: Literal["human", "ai"]
    message: str

# --- Embedded CSS for chat UI ---
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

# --- State management ---
def init_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = "user-session"
    if "session_notes" not in st.session_state:
        st.session_state.session_notes = []

# --- Per-session memory store ---
store = {}
def get_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
        store[session_id].add_ai_message(
            "Hello, I’m MindEase 🧘 – your gentle relaxation companion. "
            "What’s been weighing on your mind?"
        )
    return store[session_id]

# --- Gemini LLM setup ---
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=st.secrets["gemini_api_key"],
    temperature=0.6
)

# --- Generate rolling recap ---
def summarize_conversation():
    # Pull last 3 user messages
    user_msgs = [msg.message for msg in st.session_state.history if msg.origin == "human"]
    if not user_msgs:
        return ""
    recent = "\n".join(user_msgs[-3:])
    return f"The user recently shared:\n{recent}\n"

# --- Chat prompt with contextual memory injection ---
def build_prompt():
    memory_summary = summarize_conversation()

    return ChatPromptTemplate.from_messages([
        ("system",
         "You are MindEase, a calm, present, and empathetic AI therapist. "
         "Your goal is to help the user manage stress and feel supported. "
         "Use emotionally intelligent, mindful language. "
         "Avoid repeating advice like breathing prompts too often. "
         "Acknowledge their feelings and guide them gently. "
         + memory_summary.strip()),
        MessagesPlaceholder(variable_name="messages")
    ])

# --- Chat chain creation per submission ---
def get_chain():
    prompt = build_prompt()
    return RunnableWithMessageHistory(
        prompt | llm,
        get_session_history=get_history,
        input_messages_key="messages",
        history_messages_key="messages"
    )

# --- Message handler ---
def on_click():
    user_input = st.session_state.human_prompt
    st.session_state.history.append(Message("human", user_input))

    chain_with_history = get_chain()
    response = chain_with_history.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config={"configurable": {"session_id": st.session_state.session_id}}
    )

    st.session_state.history.append(Message("ai", response.content))

# --- UI Layout ---
init_state()
load_css()

st.title("🧘 MindEase: Gemini 1.5 Flash (Emotion-Aware)")
st.markdown("Let’s process your thoughts gently, with calm presence and continuity.")

chat_placeholder = st.container()
prompt_placeholder = st.form("chat-form")

# Display history
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

# Input field
with prompt_placeholder:
    st.markdown("**What’s on your mind?**")
    cols = st.columns((6, 1))
    cols[0].text_input("Your message", key="human_prompt", label_visibility="collapsed")
    cols[1].form_submit_button("Submit", type="primary", on_click=on_click)

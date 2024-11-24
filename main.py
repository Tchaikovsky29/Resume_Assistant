import streamlit as st
import speech_recognition as sr
from io import BytesIO
from rag_chain import rag_chain
from components import Memory, speak_text

# Initialize session state for conversation
if "conversation" not in st.session_state:
    st.session_state["conversation"] = []
if "memory" not in st.session_state:
    st.session_state["memory"] = Memory()

# Title
st.title("AI Chat Assistant")
# Chat window with scrollable container
chat_html = """
<div style="
    border: 1px solid #ccc;
    padding: 10px;
    height: 300px;
    overflow-y: auto;
    background-color: #000000;
    color: white;
    border-radius: 5px;"
    id="chat-window">
    {messages}
</div>
<script>
    const chatWindow = document.getElementById('chat-window');
    chatWindow.scrollTop = chatWindow.scrollHeight;
</script>
""".format(messages="<br><br>".join(st.session_state["conversation"]))

st.components.v1.html(chat_html, height=350)
if len(st.session_state["memory"].memory)>0:
    speak_text(st.session_state["memory"].memory[-1]["AI"])
@st.fragment
def get_text_input():
    query = ""
    query = st.text_input("Type your query here:")
    if st.button("Done"):
        if query:
            st.session_state["conversation"].append(f"<b>You:</b> {query}")
            response(query)
            st.success("sent")

@st.fragment
def get_audio_input():
    query = ""
    audio_file = st.audio_input("tap to record")
    if audio_file is not None:
        audio_bytes = BytesIO(audio_file.read())  # Create a file-like object
        recognizer = sr.Recognizer()
        try:
            with sr.AudioFile(audio_bytes) as source:
                audio = recognizer.record(source)
            query = recognizer.recognize_google(audio)
            if st.button("Send"):
                if query:
                    st.session_state["conversation"].append(f"<b>You:</b> {query}")
                    response(query)
                    st.success("sent")
        except Exception as e:
            st.error(f"Error processing the audio file: {e}")
    # AI Response
def response(query):
    response = rag_chain(query)
    st.session_state["conversation"].append(f"<b>AI:</b> {response}")
    st.rerun()
get_text_input()
get_audio_input()
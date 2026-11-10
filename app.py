
import streamlit as st
import whisper
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="VoiceText AI",
    page_icon="🎙️",
    layout="centered"
)

st.title("🎙️ VoiceText AI")
st.write("Convert your speech into text using AI.")

# Load model only once
@st.cache_resource
def load_model():
    return whisper.load_model("tiny")

model = load_model()

# Upload audio
audio_file = st.file_uploader(
    "Upload your audio",
    type=["wav", "mp3", "m4a", "ogg", "flac"]
)

if audio_file is not None:

    st.audio(audio_file)

    if st.button("🚀 Transcribe Audio"):

        # Save uploaded audio temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=os.path.splitext(audio_file.name)[1]
        ) as temp_file:

            temp_file.write(audio_file.getbuffer())
            audio_path = temp_file.name

        try:
            with st.spinner("AI is listening..."):
                result = model.transcribe(audio_path)

            text = result["text"].strip()

            st.success("Transcription completed!")

            st.subheader("Your Transcription")

            st.text_area(
                "Recognized Text",
                value=text,
                height=200
            )

            st.download_button(
                "Download Transcript",
                data=text,
                file_name="transcription.txt",
                mime="text/plain"
            )

        except Exception as error:
            st.error(f"Error: {error}")

        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

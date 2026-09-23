import io
import os
import wave
import streamlit as st
from google import genai
from google.genai import types
from google.genai.errors import APIError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Page setup
st.set_page_config(
    page_title="Gemini Article Audio Generator",
    page_icon="🎙️",
    layout="centered"
)

st.title("🎙️ Article to Verbal Soundtrack")
st.markdown("Upload or paste an article to translate and generate a spoken audio soundtrack using Google Gemini.")

# Sidebar for configuration & API key handling
st.sidebar.header("Configuration")

env_api_key = os.environ.get("GEMINI_API_KEY", "")

if env_api_key:
    api_key = env_api_key
    st.sidebar.success("Gemini API Key detected from Environment.")
else:
    api_key = st.sidebar.text_input(
        "Enter your Gemini API Key:", 
        type="password",
        help="Get your key from Google AI Studio"
    )

voice_choice = st.sidebar.selectbox(
    "Select Voice:",
    options=["Puck", "Charon", "Kore", "Fenrir", "Aoede"],
    index=0,
    help="Select the TTS voice configuration."
)

target_language = st.sidebar.selectbox(
    "Translate / Adapt To:",
    options=["Keep Original Language", "English", "Traditional Chinese", "Simplified Chinese", "Spanish", "French", "German", "Japanese"],
    index=0
)

# Text Input Methods
st.subheader("1. Input Text")
input_method = st.radio("Choose input mode:", ["Upload File (.txt)", "Paste Text Direct"])

transcript_text = ""

if input_method == "Upload File (.txt)":
    uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
    if uploaded_file is not None:
        transcript_text = uploaded_file.read().decode("utf-8")
        st.text_area("File Preview", transcript_text, height=150, disabled=True)
else:
    transcript_text = st.text_area("Paste your article text here:", height=200)


def convert_pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 24000, channels: int = 1, sample_width: int = 2) -> bytes:
    """Wraps raw PCM audio bytes with a standard RIFF WAV header."""
    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)  # 2 bytes = 16-bit PCM
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    return wav_io.getvalue()


# Retry handlers
@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(APIError),
    reraise=True
)
def translate_text(client, text, language):
    """Step 1: Translate text using gemini-2.0-flash."""
    prompt = f"Translate the following text into {language}. Return ONLY the translation, optimized for spoken reading, with no extra commentary:\n\n{text}"
    response = client.models.generate_content(
        model='gemini-3.5-flash-lite',
        contents=prompt
    )
    return response.text


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(APIError),
    reraise=True
)
def generate_audio(client, text, voice):
    """Step 2: Generate audio soundtrack using gemini-2.0-flash."""
    prompt = f"Please read out the following text clearly:\n\n{text}"
    return client.models.generate_content(
        model='gemini-3.8-flash-lite-tts',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice
                    )
                )
            )
        )
    )

st.subheader("2. Generate Audio")

if st.button("Generate Soundtrack", type="primary"):
    if not api_key:
        st.error("Please provide a valid Gemini API Key in the sidebar or via environment variables.")
    elif not transcript_text.strip():
        st.warning("Please provide some text to process.")
    else:
        try:
            client = genai.Client(api_key=api_key)

            # Step 1: Translation Phase
            if target_language != "Keep Original Language":
                with st.spinner(f"Translating article to {target_language}..."):
                    processed_text = translate_text(client, transcript_text, target_language)
                    st.text_area("Translation Preview", processed_text, height=150)
            else:
                processed_text = transcript_text

            # Step 2: Audio Generation Phase
            with st.spinner("Generating speech soundtrack..."):
                response = generate_audio(client, processed_text, voice_choice)

                raw_pcm_bytes = None

                if response.candidates and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if part.inline_data and part.inline_data.mime_type.startswith("audio/"):
                            raw_pcm_bytes = part.inline_data.data
                            break

                if raw_pcm_bytes:
                    # Convert raw PCM bytes to valid WAV format with headers
                    wav_bytes = convert_pcm_to_wav(raw_pcm_bytes, sample_rate=24000, channels=1, sample_width=2)

                    st.success("Audio soundtrack generated successfully!")

                    # Audio Player
                    st.audio(wav_bytes, format="audio/wav")

                    # Download button
                    st.download_button(
                        label="📥 Download Audio (.wav)",
                        data=wav_bytes,
                        file_name="gemini_soundtrack.wav",
                        mime="audio/wav"
                    )
                else:
                    st.error("Model responded, but no raw audio data was returned.")

        except APIError as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                st.error("Gemini servers are busy right now. Please try again in a few seconds.")
            else:
                st.error(f"API Error: {str(e)}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

# tabs/speech_emotion_recognition.py

import gradio as gr
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from transformers import pipeline
import torch
import warnings
import logging
import base64
from io import BytesIO
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress specific warnings from transformers if needed
warnings.filterwarnings("ignore", category=UserWarning, module='transformers')

# Determine the device
def get_device():
    """
    Determines the available device for inference.

    Returns:
        torch.device: The available device ('cuda', 'mps', or 'cpu').
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device for inference.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA device for inference.")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU for inference.")
    return device

device = get_device()

# Initialize the pipelines with the specified device
def initialize_pipelines(device):
    """
    Initializes the emotion and transcription pipelines.

    Args:
        device (torch.device): The device to run the pipelines on.

    Returns:
        tuple: A tuple containing the emotion_model and transcription_model.
    """
    # Gradio and transformers pipeline device handling:
    # -1: CPU
    # >=0: CUDA device index
    # 'mps' is treated as CPU since transformers may not fully support it yet
    if device.type == "cuda":
        device_index = 0  # Adjust if multiple GPUs are available
    else:
        device_index = -1  # CPU for 'cpu' and 'mps'

    try:
        emotion_model = pipeline(
            "audio-classification",
            model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
            device=device_index
        )
        logger.info("Emotion model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading emotion model: {e}")
        emotion_model = None

    try:
        transcription_model = pipeline(
            "automatic-speech-recognition",
            model="facebook/wav2vec2-base-960h",
            device=device_index
        )
        logger.info("Transcription model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading transcription model: {e}")
        transcription_model = None

    return emotion_model, transcription_model

emotion_model, transcription_model = initialize_pipelines(device)

# Emotion Mapping
# Ensure that these labels match exactly with the emotion_model's output labels
emotion_mapping = {
    "angry": (0.8, 0.8, -0.5),
    "happy": (0.6, 0.6, 0.8),
    "sad": (-0.6, -0.4, -0.6),
    "neutral": (0, 0, 0),
    "fear": (0.3, -0.3, -0.7),
    "surprise": (0.4, 0.2, 0.2),
    "disgust": (0.2, 0.5, -0.6)
    # Add or remove emotions based on the model's actual labels
}

def plot_to_pil(plt_fig):
    """
    Converts a matplotlib figure to a PIL Image.

    Args:
        plt_fig (matplotlib.figure.Figure): The matplotlib figure to convert.

    Returns:
        PIL.Image.Image: The converted PIL Image.
    """
    buf = BytesIO()
    plt_fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf).convert('RGB')  # Ensure RGB
    plt.close(plt_fig)
    return img

def process_audio_emotion(audio_file):
    """
    Processes the input audio file to perform transcription and emotion recognition.
    Generates waveform and mel spectrogram plots.

    Args:
        audio_file (str): The filepath to the uploaded audio file.

    Returns:
        tuple: A tuple containing:
            - Transcription (str)
            - Emotion (str)
            - Confidence (%) (float)
            - Arousal (float)
            - Dominance (float)
            - Valence (float)
            - Waveform Plot (PIL.Image.Image)
            - Mel Spectrogram Plot (PIL.Image.Image)
    """
    if not audio_file:
        return (
            "No audio file provided.",  # Transcription (textbox)
            "N/A",                       # Emotion (textbox)
            0.0,                         # Confidence (%) (number)
            0.0,                         # Arousal (number)
            0.0,                         # Dominance (number)
            0.0,                         # Valence (number)
            None,                        # Waveform Plot (image)
            None                         # Mel Spectrogram Plot (image)
        )

    try:
        y, sr = librosa.load(audio_file, sr=None)

        # Transcription
        if transcription_model:
            transcription_result = transcription_model(audio_file)
            transcription = transcription_result.get("text", "N/A")
        else:
            transcription = "Transcription model not loaded."

        # Emotion Recognition
        if emotion_model:
            emotion_results = emotion_model(audio_file)
            if emotion_results:
                emotion_result = emotion_results[0]
                emotion = emotion_result.get("label", "Unknown").lower()
                confidence = emotion_result.get("score", 0.0) * 100  # Convert to percentage
                arousal, dominance, valence = emotion_mapping.get(emotion, (0.0, 0.0, 0.0))
            else:
                emotion = "No emotion detected."
                confidence = 0.0
                arousal, dominance, valence = 0.0, 0.0, 0.0
        else:
            emotion = "Emotion model not loaded."
            confidence = 0.0
            arousal, dominance, valence = 0.0, 0.0, 0.0

        # Plotting Waveform
        fig_waveform, ax_waveform = plt.subplots(figsize=(10, 4))
        librosa.display.waveshow(y, sr=sr, ax=ax_waveform)
        ax_waveform.set_title("Waveform")
        ax_waveform.set_xlabel("Time (s)")
        ax_waveform.set_ylabel("Amplitude")
        waveform_plot_pil = plot_to_pil(fig_waveform)

        # Plotting Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        fig_mel, ax_mel = plt.subplots(figsize=(10, 4))
        S_dB = librosa.power_to_db(mel_spec, ref=np.max)
        img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax_mel)
        fig_mel.colorbar(img, ax=ax_mel, format='%+2.0f dB')
        ax_mel.set_title("Mel Spectrogram")
        mel_spec_plot_pil = plot_to_pil(fig_mel)

        return (
            transcription,                  # Transcription (textbox)
            emotion.capitalize(),           # Emotion (textbox)
            confidence,                     # Confidence (%) (number)
            arousal,                        # Arousal (number)
            dominance,                      # Dominance (number)
            valence,                        # Valence (number)
            waveform_plot_pil,              # Waveform Plot (image)
            mel_spec_plot_pil               # Mel Spectrogram Plot (image)
        )
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        return (
            f"Error: {str(e)}",  # Transcription (textbox)
            "Error",              # Emotion (textbox)
            0.0,                  # Confidence (%) (number)
            0.0,                  # Arousal (number)
            0.0,                  # Dominance (number)
            0.0,                  # Valence (number)
            None,                 # Waveform Plot (image)
            None                  # Mel Spectrogram Plot (image)
        )

def create_emotion_recognition_tab():
    """
    Creates the Emotion Recognition tab in the Gradio interface.
    """
    with gr.Blocks() as demo:
        gr.Markdown("## Speech Emotion Recognition")

        with gr.Row():
            with gr.Column(scale=2):
                input_audio = gr.Audio(label="Input Audio", type="filepath")
                gr.Examples(
                    examples=["./assets/audio/fitness.wav"],
                    inputs=[input_audio],
                    label="Examples"
                )
            with gr.Column(scale=1):
                transcription_output = gr.Textbox(label="Transcription", interactive=False)
                emotion_output = gr.Textbox(label="Emotion", interactive=False)
                confidence_output = gr.Number(label="Confidence (%)", interactive=False)
                arousal_output = gr.Number(label="Arousal (Level of Energy)", interactive=False)
                dominance_output = gr.Number(label="Dominance (Degree of Control)", interactive=False)
                valence_output = gr.Number(label="Valence (Positivity/Negativity)", interactive=False)
            with gr.Column(scale=2):
                waveform_plot = gr.Image(label="Waveform")
                mel_spec_plot = gr.Image(label="Mel Spectrogram")

        # Define the interaction: when audio changes, process it
        input_audio.change(
            fn=process_audio_emotion,
            inputs=[input_audio],
            outputs=[
                transcription_output,
                emotion_output,
                confidence_output,
                arousal_output,
                dominance_output,
                valence_output,
                waveform_plot,
                mel_spec_plot
            ]
        )

    return demo
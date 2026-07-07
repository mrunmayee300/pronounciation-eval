import streamlit as st
from faster_whisper import WhisperModel
import torch
from typing import Tuple, List, Dict, Any

@st.cache_resource
def load_whisper_model(model_size: str = "base") -> WhisperModel:
    """
    Loads and caches the faster-whisper model.
    Checks for GPU acceleration (CUDA) automatically, falling back to CPU.
    Uses int8 quantization on CPU and float16 on GPU for efficiency.
    """
    if torch.cuda.is_available():
        device = "cuda"
        compute_type = "float16"
    else:
        device = "cpu"
        compute_type = "int8"

    # Load CTranslate2-based Whisper model
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    return model

def transcribe_audio(
    model: WhisperModel, 
    audio_path: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Any]:
    """
    Transcribes English audio and returns word-level detail and segment-level details.
    
    Returns:
        words: List of dicts, each with keys {"word", "start", "end", "confidence"}
        segments: List of dicts, each with keys {"id", "text", "start", "end", "no_speech_prob", "avg_logprob"}
        info: Transcription info metadata from faster-whisper
    """
    # Force language="en" since the evaluator is specifically for English pronunciation.
    # Enable word_timestamps to get individual word timing and confidence scores.
    segments_generator, info = model.transcribe(
        audio_path,
        beam_size=5,
        word_timestamps=True,
        language="en"
    )

    words = []
    segments = []

    # Iterate through segments generator to perform the transcription
    for segment in segments_generator:
        seg_detail = {
            "id": segment.id,
            "text": segment.text.strip(),
            "start": segment.start,
            "end": segment.end,
            "no_speech_prob": segment.no_speech_prob,
            "avg_logprob": segment.avg_logprob
        }
        segments.append(seg_detail)

        # Extract word timestamps if available
        if segment.words is not None:
            for w in segment.words:
                words.append({
                    "word": w.word.strip(),
                    "start": w.start,
                    "end": w.end,
                    "confidence": w.probability
                })
        else:
            # Robust fallback: if Whisper fails to output word-level timestamps,
            # split segment text and distribute duration evenly with a default confidence.
            seg_words = segment.text.strip().split()
            if seg_words:
                count = len(seg_words)
                duration = segment.end - segment.start
                word_duration = duration / count if count > 0 else 0
                for idx, w_text in enumerate(seg_words):
                    words.append({
                        "word": w_text,
                        "start": segment.start + (idx * word_duration),
                        "end": segment.start + ((idx + 1) * word_duration),
                        "confidence": 0.80  # Default neutral confidence
                    })

    return words, segments, info

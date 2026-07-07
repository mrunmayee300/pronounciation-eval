import os
import string
from typing import List, Dict, Any

def ensure_directories() -> None:
    """
    Ensures that the required uploads/ and resources/ directories exist.
    """
    for dir_name in ["uploads", "resources"]:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

def render_highlighted_transcript(
    words: List[Dict[str, Any]], 
    low_confidence_threshold: float
) -> str:
    """
    Generates an HTML string of the transcript with highlighted words.
    - Low-confidence words are highlighted in red with a dashed underline.
    - Hesitation/filler words are highlighted in orange/yellow with a dotted underline.
    The layout uses CSS styles that look premium and work across Streamlit themes.
    """
    from utils.scoring import HESITATION_FILLERS, clean_word

    html_tokens = []
    
    for w in words:
        word_text = w["word"]
        cleaned = clean_word(word_text)
        
        # Determine highlighting based on scoring classification
        if cleaned in HESITATION_FILLERS:
            # Orange warning highlight for hesitations
            token_html = (
                f'<span style="background-color: rgba(243, 156, 18, 0.15); '
                f'color: #d35400; border-bottom: 2px dotted #d35400; '
                f'padding: 2px 4px; margin: 0 1px; border-radius: 4px; '
                f'font-style: italic; font-weight: 500;" '
                f'title="Hesitation/Filler Word">{word_text}</span>'
            )
        elif w["confidence"] < low_confidence_threshold:
            # Red/crimson highlight for poor pronunciation
            token_html = (
                f'<span style="background-color: rgba(211, 47, 47, 0.12); '
                f'color: #c62828; border-bottom: 2px dashed #c62828; '
                f'padding: 2px 4px; margin: 0 1px; border-radius: 4px; '
                f'font-weight: bold;" '
                f'title="Low Confidence: {w["confidence"] * 100:.0f}%">{word_text}</span>'
            )
        else:
            # Standard output
            token_html = f'<span style="padding: 2px 0; margin: 0 1px;">{word_text}</span>'
            
        html_tokens.append(token_html)

    # Wrap in a nicely padded container div
    container_html = (
        f'<div style="font-size: 1.15rem; line-height: 2.0; '
        f'font-family: inherit; padding: 1.5rem; border-radius: 8px; '
        f'border: 1px solid rgba(128, 128, 128, 0.2); '
        f'background-color: rgba(128, 128, 128, 0.05); margin-bottom: 1.5rem;">'
        f'{" ".join(html_tokens)}'
        f'</div>'
    )
    
    return container_html

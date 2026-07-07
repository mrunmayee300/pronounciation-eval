import streamlit as st
from openai import OpenAI
from transformers import pipeline
import json
import string
from typing import List, Dict, Any

# Local dictionary of common mispronounced words and their phonetic advice
FALLBACK_ADVICE = {
    "the": {
        "issue": "Difficulty with the voiced dental fricative '/ð/' sound.",
        "suggested_improvement": "Place the tip of your tongue behind or between your front teeth, blow air, and make your vocal cords vibrate."
    },
    "three": {
        "issue": "Voiceless dental fricative '/θ/' combined with '/r/'.",
        "suggested_improvement": "Start with your tongue between your teeth for the 'th' sound, then pull it back quickly without touching the roof of your mouth for the 'r'."
    },
    "with": {
        "issue": "Ending with dental fricative '/ð/' or '/θ/'.",
        "suggested_improvement": "Do not pronounce it as a 't' or 'd'. Let air flow through your teeth with the tongue lightly touching them."
    },
    "world": {
        "issue": "Tricky consonant blend of '/r/', '/l/', and '/d/'.",
        "suggested_improvement": "Break it into two beats: say 'were' first, then transition to a soft 'ld' sound by curling your tongue up."
    },
    "comfortable": {
        "issue": "Syllable stress and silent letters (often mispronounced as com-for-ta-ble).",
        "suggested_improvement": "Pronounce it as three syllables: 'KUMF-ter-bul'. Stress the first syllable and keep the second very short."
    },
    "ask": {
        "issue": "Syllable metathesis (swapping 's' and 'k' sounds to sound like 'ax').",
        "suggested_improvement": "Focus on pronouncing the continuous 's' (hissing sound) before the sharp release of the 'k' sound."
    },
    "specifically": {
        "issue": "Multisyllabic rhythm and consonant cluster 'sp' and 'cally'.",
        "suggested_improvement": "Break into beats: 'spe-CIF-ic-lee'. Stress the second syllable ('CIF') and say the rest quickly."
    },
    "literature": {
        "issue": "Elision of syllables and pronunciation of the final '-ture'.",
        "suggested_improvement": "Say 'LIT-er-a-chur'. The ending sounds like 'chur' (as in chair), with stress on the very first syllable."
    },
    "uncomfortable": {
        "issue": "Syllable stress on 'comf' and vowel reduction.",
        "suggested_improvement": "Pronounce it as 'un-KUMF-ter-bul'. Keep the middle syllables quick and focus stress on 'KUMF'."
    },
    "library": {
        "issue": "Often mispronounced by omitting the first 'r' ('liberry').",
        "suggested_improvement": "Make sure to pronounce both 'r's clearly: 'LY-brar-ee'. Do not drop the middle 'r'."
    },
    "probably": {
        "issue": "Slurring or omitting the middle syllable ('probly').",
        "suggested_improvement": "Say all three syllables clearly: 'PROB-ab-lee'. Focus on the middle 'ab' sound."
    },
    "scheduled": {
        "issue": "The initial 'sch' sound can vary between '/sk/' (US) and '/ʃ/' (UK).",
        "suggested_improvement": "In US English, pronounce it as 'SKEJ-uld'. In UK English, it starts as 'SHEJ-uld'. Ensure a clear 'd' ending."
    },
    "months": {
        "issue": "The difficult 'nths' consonant cluster.",
        "suggested_improvement": "Make a clear nasal 'n', quickly touch the tip of the tongue to your teeth for 'th', and finish with a hissing 's'."
    },
    "clothes": {
        "issue": "The combination of voiced '/ð/' and '/z/'. It is often simplified to 'kloz'.",
        "suggested_improvement": "Pronouncing it as 'kloz' (rhymes with close/nose) is widely accepted. To be formal, pronounce a tiny 'th' before the 'z'."
    },
    "focus": {
        "issue": "Vowel pronunciation of the 'o' and the final 'us' syllable.",
        "suggested_improvement": "Ensure a long 'o' sound: 'FOH-kus'. Do not make the second syllable sound like 'cuss' with too much force."
    }
}

@st.cache_resource
def load_hf_pipeline(model_name: str = "google/bert_uncased_L-2_H-128_A-2") -> Any:
    """
    Loads and caches a HuggingFace fill-mask pipeline.
    prajjwal1/bert-tiny is extremely lightweight (~17MB) and loads instantly on CPU.
    """
    try:
        return pipeline("fill-mask", model=model_name)
    except Exception as e:
        try:
            st.sidebar.warning(f"Failed to load HuggingFace model '{model_name}': {str(e)}. Using fallback suggestions.")
        except Exception:
            pass
        return None


def get_hf_context_suggestions(
    hf_pipeline: Any, 
    words: List[Dict[str, Any]], 
    flagged_word_dict: Dict[str, Any], 
    window_size: int = 5
) -> List[str]:
    """
    Constructs a localized window of text around the flagged word, replaces it with [MASK],
    and queries the HuggingFace fill-mask pipeline to suggest 3 grammatically suitable alternatives.
    """
    if hf_pipeline is None:
        return []

    idx = flagged_word_dict["index"]
    mask_token = hf_pipeline.tokenizer.mask_token

    # Build surrounding word context window
    start_win = max(0, idx - window_size)
    end_win = min(len(words), idx + window_size + 1)

    context_tokens = []
    for i in range(start_win, end_win):
        if i == idx:
            context_tokens.append(mask_token)
        else:
            context_tokens.append(words[i]["word"])

    masked_sentence = " ".join(context_tokens)

    try:
        predictions = hf_pipeline(masked_sentence)
        
        # Standardize predictions format (can be dict if single suggestion, but usually list of dicts)
        if isinstance(predictions, dict):
            predictions = [predictions]
            
        suggestions = []
        original_lower = flagged_word_dict["word"].lower().strip(string.punctuation)
        
        for pred in predictions:
            token = pred["token_str"].strip().lower()
            token = token.strip(string.punctuation)
            
            # Simple filters: ignore tiny non-word tokens (except single letters like a/i) and duplicates
            if len(token) > 1 or token in {"a", "i"}:
                if token not in suggestions and token != original_lower:
                    suggestions.append(token)
                    
            if len(suggestions) >= 3:
                break
                
        return suggestions
    except Exception:
        return []

def get_local_fallback_advice(word_str: str) -> Dict[str, str]:
    """
    Rule-based engine providing phonetics-based advice for words not present in the hardcoded list.
    """
    w_lower = word_str.lower().strip(string.punctuation)

    # Rule-based spelling matchers
    if "th" in w_lower:
        advice = {
            "issue": "Potential challenge with dental fricative ('th') sound.",
            "suggested_improvement": "Place your tongue tip between your front teeth. Blow air for voiceless 'th' (e.g. 'think') or add vocal vibration for voiced 'th' (e.g. 'this')."
        }
    elif "r" in w_lower:
        advice = {
            "issue": "Rhotic '/r/' articulation.",
            "suggested_improvement": "Curl the sides of your tongue against your upper molars. The tip of your tongue should be suspended, not touching the roof of your mouth."
        }
    elif "l" in w_lower:
        advice = {
            "issue": "L-sound placement (light vs. dark L).",
            "suggested_improvement": "For a clear 'L', press the tip of your tongue flatly against the gum ridge behind your upper front teeth."
        }
    elif w_lower.endswith("ing"):
        advice = {
            "issue": "Weak production of the final nasal '/ŋ/' sound.",
            "suggested_improvement": "Ensure you push the back of your tongue against your soft palate in the throat to make the 'ng' sound, rather than letting it slide to a standard dental 'n'."
        }
    elif "sp" in w_lower or "st" in w_lower or "sk" in w_lower:
        advice = {
            "issue": "Consonant cluster blend articulation.",
            "suggested_improvement": "Ensure you flow from the continuous 's' to the stop consonant without inserting a helper vowel (like 'es-') before the word."
        }
    elif len(w_lower) >= 9:
        advice = {
            "issue": "Complex multisyllabic rhythm or word stress.",
            "suggested_improvement": "Slow down and speak in beats. Determine the stressed syllable and make its vowel sound longer, louder, and higher in pitch."
        }
    else:
        advice = {
            "issue": "General vowel or consonant articulation clarity.",
            "suggested_improvement": "Pronounce the word slowly. Pay attention to jaw opening on vowels and make sure to fully enunciate ending stop-consonants like 't', 'd', or 'k'."
        }
        
    advice["word"] = word_str
    return advice


def generate_advice(flagged_words: List[Dict[str, Any]], openai_key: str) -> List[Dict[str, str]]:
    """
    Generates pronunciation feedback advice for flagged words.
    Queries OpenAI API if a key is provided; otherwise, falls back to the robust local phonetic engine.
    """
    if not flagged_words:
        return []

    # Fallback mode if no API key is specified
    if not openai_key or not openai_key.strip():
        feedback_list = []
        for w in flagged_words:
            word_cleaned = w["word"].lower().strip(string.punctuation)
            if word_cleaned in FALLBACK_ADVICE:
                advice = FALLBACK_ADVICE[word_cleaned].copy()
                advice["word"] = w["word"]
            else:
                advice = get_local_fallback_advice(w["word"])
            feedback_list.append(advice)
        return feedback_list

    # OpenAI API Execution
    try:
        client = OpenAI(api_key=openai_key)
        words_input = [w["word"] for w in flagged_words]
        
        prompt = f"""
        You are an expert English pronunciation coach. 
        The following English words were flagged as mispronounced or unclear in an audio recording:
        {json.dumps(words_input)}
        
        Analyze each word and provide concise, specific pronunciation advice.
        For each word, describe:
        1. The likely "issue" (e.g., typical phonetic substitution, consonant blend challenge, vowel reduction, or silent letter omission).
        2. A "suggested_improvement" containing a practical physical tip (e.g., mouth positioning, tongue placement).
        
        You MUST respond with a valid JSON array of objects, where each object has these exact keys:
        "word": the original word string
        "issue": short description of the pronunciation challenge (max 1 sentence)
        "suggested_improvement": actionable vocal coaching tip (max 2 sentences)
        
        Do NOT wrap the output in markdown code blocks. Just return raw JSON.
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional ESL phonetic coach who speaks only in valid JSON arrays."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        content = response.choices[0].message.content.strip()
        
        # Clean potential markdown output formatting
        if content.startswith("```"):
            content = content.replace("```json", "").replace("```", "").strip()

        data = json.loads(content)

        # Parse and standardize output
        if isinstance(data, dict):
            for k in ["feedback", "words", "data"]:
                if k in data and isinstance(data[k], list):
                    data = data[k]
                    break

        if isinstance(data, list):
            # Clean structure and return
            sanitized = []
            for item in data:
                if "word" in item and "issue" in item and "suggested_improvement" in item:
                    sanitized.append({
                        "word": item["word"],
                        "issue": item["issue"],
                        "suggested_improvement": item["suggested_improvement"]
                    })
            if len(sanitized) > 0:
                return sanitized

        # Fallback if parsed format is invalid
        raise ValueError("Invalid format received from OpenAI API.")

    except Exception as e:
        # Graceful fallback on API error/rate-limit/network issues
        feedback_list = []
        for w in flagged_words:
            word_cleaned = w["word"].lower().strip(string.punctuation)
            if word_cleaned in FALLBACK_ADVICE:
                advice = FALLBACK_ADVICE[word_cleaned].copy()
                advice["word"] = w["word"]
            else:
                advice = get_local_fallback_advice(w["word"])
            feedback_list.append(advice)
        return feedback_list

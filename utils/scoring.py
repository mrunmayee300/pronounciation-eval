import string
from typing import List, Dict, Any

# Common hesitation filler words in English
HESITATION_FILLERS = {"uh", "um", "ah", "er", "eh", "hm", "hmm", "like"}

def clean_word(word_str: str) -> str:
    """
    Strips punctuation from a word and converts it to lowercase.
    This ensures accurate matching for hesitation detection.
    """
    return word_str.lower().strip(string.punctuation)

def calculate_pronunciation_score(
    words: List[Dict[str, Any]], 
    segments: List[Dict[str, Any]], 
    low_confidence_threshold: float = 0.7,
    audio_duration_seconds: float = 35.0
) -> Dict[str, Any]:
    """
    Computes a pronunciation score out of 100 based on word confidence and fluency heuristics.
    Also computes enhanced IELTS/PTE-style benchmark subscores:
      - Accuracy Score: based on average confidence of content words
      - Fluency Score: based on speech rate (WPM) and hesitation penalties
      - Clarity Score: based on speech clarity and segment noise penalties
      - Enhanced Overall Score: weighted average of the three subscores
    """
    score = 100
    flagged_words = []
    hesitation_words = []
    unclear_segments = []

    # 1. Evaluate individual words for confidence and filler word usage
    content_word_confidences = []

    for idx, w in enumerate(words):
        cleaned = clean_word(w["word"])
        
        # We will focus on traditional fillers: "uh", "um", "ah", "er", "eh", "hm", "hmm"
        traditional_fillers = {"uh", "um", "ah", "er", "eh", "hm", "hmm"}
        if cleaned in traditional_fillers:
            hesitation_words.append({
                "word": w["word"],
                "start": w["start"],
                "end": w["end"],
                "confidence": w["confidence"],
                "index": idx
            })
        else:
            content_word_confidences.append(w["confidence"])
            if w["confidence"] < low_confidence_threshold:
                flagged_words.append({
                    "word": w["word"],
                    "start": w["start"],
                    "end": w["end"],
                    "confidence": w["confidence"],
                    "index": idx
                })

    # 2. Check segments for clarity
    for seg in segments:
        # Get all words that belong to this segment to calculate average confidence
        seg_words = [
            w for w in words
            if w["start"] >= seg["start"] - 0.05 and w["end"] <= seg["end"] + 0.05
        ]
        
        avg_confidence = (
            sum(w["confidence"] for w in seg_words) / len(seg_words) 
            if seg_words else 1.0
        )

        is_unclear = False
        reason = ""

        # Flag unclear segments based on silence probability or low word confidence
        if seg["no_speech_prob"] > 0.5:
            is_unclear = True
            reason = f"High background noise/no speech probability ({seg['no_speech_prob'] * 100:.1f}%)"
        elif avg_confidence < 0.5:
            is_unclear = True
            reason = f"Low segment average confidence ({avg_confidence * 100:.1f}%)"

        if is_unclear:
            unclear_segments.append({
                "text": seg["text"],
                "start": seg["start"],
                "end": seg["end"],
                "reason": reason,
                "no_speech_prob": seg["no_speech_prob"],
                "avg_confidence": avg_confidence
            })

    # 3. Calculate assignment score deductions
    low_conf_deduction = len(flagged_words) * 3
    hesitation_deduction = len(hesitation_words) * 2
    unclear_segment_deduction = len(unclear_segments) * 2

    total_deductions = low_conf_deduction + hesitation_deduction + unclear_segment_deduction
    score -= total_deductions
    score = max(0, min(100, score))

    # 4. ENHANCED BENCHMARK METRICS
    
    # Accuracy Subscore: Based on average confidence of content words
    if content_word_confidences:
        base_accuracy = (sum(content_word_confidences) / len(content_word_confidences)) * 100
        # Penalize severely low confidence content words (e.g. < 0.40) to reflect articulation flaws
        severe_count = sum(1 for c in content_word_confidences if c < 0.40)
        accuracy_score = base_accuracy - (severe_count * 2)
    else:
        accuracy_score = 100
    accuracy_score = max(0, min(100, accuracy_score))

    # Speech Rate (Words Per Minute)
    total_non_filler_words = len(content_word_confidences)
    if audio_duration_seconds > 0:
        wpm = (total_non_filler_words / audio_duration_seconds) * 60
    else:
        wpm = 0

    # Speech Rate Status & Deductions
    wpm_deduction = 0
    if wpm < 80:
        wpm_status = "Too Slow (Hesitant)"
        wpm_deduction = min(20, (80 - wpm) * 0.5)
    elif wpm < 110:
        wpm_status = "Slow (Deliberate)"
        wpm_deduction = min(10, (110 - wpm) * 0.25)
    elif wpm <= 160:
        wpm_status = "Excellent (Fluent Speaking Rate)"
        wpm_deduction = 0
    elif wpm <= 180:
        wpm_status = "Fast (Rushed)"
        wpm_deduction = min(10, (wpm - 160) * 0.25)
    else:
        wpm_status = "Too Fast (Unnatural)"
        wpm_deduction = min(20, (wpm - 180) * 0.5)

    # Fluency Subscore: Filler pauses + Speaking rate deviations
    fluency_score = 100 - (len(hesitation_words) * 5) - wpm_deduction
    fluency_score = max(0, min(100, fluency_score))

    # Clarity Subscore: Segment noise/mumbling penalties
    clarity_score = 100 - (len(unclear_segments) * 15)
    clarity_score = max(0, min(100, clarity_score))

    # Weighted Enhanced Score: 50% Accuracy, 30% Fluency, 20% Clarity
    enhanced_overall = (accuracy_score * 0.5) + (fluency_score * 0.3) + (clarity_score * 0.2)
    enhanced_overall = max(0, min(100, enhanced_overall))

    return {
        "score": int(score),
        "total_deductions": total_deductions,
        "low_conf_deduction": low_conf_deduction,
        "hesitation_deduction": hesitation_deduction,
        "unclear_segment_deduction": unclear_segment_deduction,
        "flagged_words": flagged_words,
        "hesitation_words": hesitation_words,
        "unclear_segments": unclear_segments,
        
        # Enhanced properties
        "accuracy_score": int(round(accuracy_score)),
        "fluency_score": int(round(fluency_score)),
        "clarity_score": int(round(clarity_score)),
        "enhanced_score": int(round(enhanced_overall)),
        "wpm": int(round(wpm)),
        "wpm_status": wpm_status
    }

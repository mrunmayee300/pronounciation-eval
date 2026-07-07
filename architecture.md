# Architectural Design: English Pronunciation Evaluator

This document details the system design, components, models, scoring algorithms, and data privacy implementation for the English Pronunciation Evaluator.

---

## 1. System Components

The application is built as a single, modular Streamlit app running on a python runtime. It avoids microservice overhead, making it easy for students to explain during interviews.

```
+-------------------------------------------------------------+
|                        Streamlit UI                         |
|             (File Upload / Mic Audio Recording)            |
+------------------------------+------------------------------+
                               |
                               | Audio File
                               v
+------------------------------+------------------------------+
|                        Audio Utility                        |
|        - Duration check using ffprobe (30s - 45s)           |
|        - Conversion to Mono 16kHz WAV using FFmpeg          |
+------------------------------+------------------------------+
                               |
                               | Standardized 16kHz WAV
                               v
+------------------------------+------------------------------+
|                    Speech Recognition Engine                |
|       - CTranslate2 faster-whisper base model (local)        |
|       - Extracts word-level timings & confidence scores      |
+------------------------------+------------------------------+
                               |
                               | Word timings & confidence
                               v
+------------------------------+------------------------------+
|                 Pronunciation Scoring Engine                |
|       - Computes metric out of 100 based on heuristics       |
|       - Identifies flagged words, fillers, noisy segments    |
+------------------------------+------------------------------+
                               |
                               | Flagged Words List
                               v
+------------------------------+------------------------------+
|                      Feedback Generator                      |
|                                                             |
|   +--------------------------+--------------------------+   |
|   |  Grammar Suggester (HF)  |   Articulatory Coach     |   |
|   |  - bert-tiny fill-mask   |   - OpenAI GPT-4o-mini   |   |
|   |  - local context fills   |     OR local rule engine |   |
|   +--------------------------+--------------------------+   |
+-------------------------------------------------------------+
```

---

## 2. Models Used and Rationale

1. **Speech Transcription: `faster-whisper` (base model)**
   - **Why**: OpenAI's Whisper model is state-of-the-art for speech-to-text. `faster-whisper` is a re-implementation of Whisper using CTranslate2, which is up to 4x faster and uses less memory than the original huggingface implementation. The `base` model (145M parameters) is small enough to run rapidly on standard CPUs, yet provides accurate word-level transcription and confidence probabilities.
2. **Contextual Vocabulary Suggestions: `google/bert_uncased_L-2_H-128_A-2` (local HuggingFace)**
   - **Why**: Used for fill-mask suggestions to show learners what other words would fit in the sentence where they had pronunciation difficulties. This model is extremely lightweight (~17MB) and loads instantly, preventing the app from crashing due to Out-Of-Memory (OOM) errors. It includes the standard `model_type` keys required by newer versions of the HuggingFace transformers framework.
3. **Pronunciation Coaching: OpenAI `gpt-4o-mini` API (with Local Fallback)**
   - **Why**: Converts raw spelling errors into friendly articulatory tips (mouth positioning/tongue placement). We only send the small subset of *flagged words* to minimize token costs, latency, and dependency. If the API key is missing or fails, the app uses a rich local phonetic dictionary and rule engine.

---

The app implements two parallel scoring tracks: a primary **Assignment Heuristic Score** (complying strictly with prompt guidelines) and a detailed **ESL Diagnostic Benchmark Score** mapping to international test standards (IELTS/PTE).

### 3.1 Primary Assignment Heuristic Score
The overall metric starts at 100 and applies simple linear deductions:

$$\text{Assignment Score} = \max\left(0, \min\left(100, 100 - D_{\text{words}} - D_{\text{hesitations}} - D_{\text{clarity}}\right)\right)$$

*   **Low-Confidence Deductions ($D_{\text{words}}$)**: Subtracts **3 points** for every content word transcribed with a confidence probability below the user-defined threshold (default `0.70`).
*   **Fluency Deductions ($D_{\text{hesitations}}$)**: Subtracts **2 points** for every verbal pause or hesitation word detected (`"uh"`, `"um"`, `"ah"`, `"er"`, `"eh"`, `"hm"`, `"hmm"`, `"like"`).
*   **Acoustic Clarity Deductions ($D_{\text{clarity}}$)**: Subtracts **2 points** for every segment where the background noise is too high (Whisper `no_speech_prob` > 0.5) or the average word confidence is very low (< 0.5).

### 3.2 ESL Diagnostic Subscores (IELTS/PTE Alignment)
To provide deep articulation diagnostics, the app computes three normalized benchmark scores (out of 100):

1.  **Accuracy Subscore**: Measures clarity of speech sounds. Derived directly from the average confidence of all content words:
    $$\text{Accuracy} = \text{Average Confidence (non-fillers)} \times 100 - 2 \times N_{\text{severe\_errors}}$$
    *(Severe errors defined as content words with Whisper confidence < 0.40)*
2.  **Fluency Subscore**: Evaluates the pacing and pauses. Starts at 100, deducts **5 points** per hesitation filler, and applies scaling penalties for Speaking Speed deviations from native speech rates:
    *   *Speaking Speed (WPM)* is calculated as: $\text{WPM} = \frac{\text{Content Words}}{\text{Audio Duration in Seconds}} \times 60$
    *   *Normal Speaking Pace*: $110 - 160$ WPM (0 penalty).
    *   *Slow Speaking Pace* ($< 110$ WPM): Deducts up to 20 points ($0.25$ to $0.5$ points per WPM deviation) to reflect hesitation gaps.
    *   *Fast Speaking Pace* ($> 160$ WPM): Deducts up to 20 points ($0.25$ to $0.5$ points per WPM deviation) to reflect rushed articulation.
3.  **Clarity Subscore**: Measures speech isolation and recording acoustics. Starts at 100 and deducts **15 points** for each segment flagged with low average confidence or high noise probability.
4.  **Overall Enhanced Score**: Computed as a weighted average:
    $$\text{Overall Enhanced} = 0.50 \times \text{Accuracy} + 0.30 \times \text{Fluency} + 0.20 \times \text{Clarity}$$

---

## 4. Highlighting Logic

The transcript display uses HTML rendering supported natively in Streamlit:
*   **Low Confidence Words**: Wrapped in red tags (`background-color: rgba(211, 47, 47, 0.12); color: #c62828; border-bottom: 2px dashed #c62828;`) with a tooltip displaying the exact Whisper confidence percentage.
*   **Hesitation Fillers**: Wrapped in orange tags (`background-color: rgba(243, 156, 18, 0.15); color: #d35400; border-bottom: 2px dotted #d35400;`) with a tooltip indicating a verbal pause.

---

## 5. DPDP Act Compliance

This app is fully aligned with the Digital Personal Data Protection (DPDP) Act:
1. **Explicit Consent**: A mandatory checkbox forces users to consent to voice processing before the analysis button is enabled.
2. **Purpose Limitation**: Audio is processed solely for scoring pronunciation and is never used for voice profiling.
3. **No Storage & Strict Deletion**:
   - Audio is saved temporarily in `uploads/` using randomized filenames (UUIDs).
   - In `app.py`, a `try...finally` block guarantees that the temporary input file and converted WAV file are **always** deleted immediately after transcription, even if the processing fails or crashes.
4. **Data Residency**: Local transcription (Whisper, HF) keeps data processing local. External transmission is limited to sending raw text strings (flagged words) to OpenAI for translation into phonetic feedback, protecting voice biometric privacy.

---

## 6. System Trade-offs

| Design Choice | Strengths | Trade-offs |
| :--- | :--- | :--- |
| **CTranslate2 Whisper (Local)** | Fast CPU execution, free, high quality transcript. | Model loading consumes initial startup latency (~5-10s on first load). |
| **Rule-Based fallback for AI** | 100% reliable, zero API costs, runs offline. | Coaching advice is pre-defined or templated compared to custom LLM feedback. |
| **No Database Architecture** | Simple, stateless, easy deployment on Streamlit Cloud. | No historical tracking of user score improvements. |

---

## 7. Future Improvements

1. **Phoneme-level Force Alignment**: Use Wav2Vec2 or Montreal Forced Aligner (MFA) to match speaker phonemes against native phoneme templates for precise sub-word grading.
2. **History & Analytics**: Add a lightweight SQLite database to allow students to track their scores over time and visualize their progress.
3. **Interactive Audio Playback**: Allow the user to click flagged words in the table to play back that specific slice of audio using Whisper timestamp indexes.

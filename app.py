import streamlit as st
import os
import uuid
import pandas as pd
from utils.helpers import ensure_directories, render_highlighted_transcript
from utils.audio import validate_audio_duration, convert_to_wav, cleanup_file
from utils.whisper import load_whisper_model, transcribe_audio
from utils.scoring import calculate_pronunciation_score
from utils.feedback import load_hf_pipeline, get_hf_context_suggestions, generate_advice
from audio_recorder_streamlit import audio_recorder

# Ensure directories exist
ensure_directories()

# Set Page Config
st.set_page_config(
    page_title="English Pronunciation Evaluator",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App Title & Subtitle
st.title("🎙️ English Pronunciation Evaluator")
st.markdown(
    "A clean, simple tool for students and language learners to assess their spoken English. "
    "Upload or record audio (between **30 and 45 seconds** long) to receive a pronunciation score, "
    "see a highlighted transcript of your speech, and get phonetic coaching advice."
)

# ----------------- SIDEBAR CONFIGURATIONS -----------------
st.sidebar.header("🔧 Settings & Keys")

# OpenAI API Key Input
openai_api_key = st.sidebar.text_input(
    "OpenAI API Key (Optional)",
    type="password",
    value=os.environ.get("OPENAI_API_KEY", ""),
    help="Used to generate personalized pronunciation advice. If empty, the app will fall back to a local phonetic dictionary."
)

if openai_api_key:
    st.sidebar.success("✨ OpenAI API enabled for feedback.")
else:
    st.sidebar.info("💡 Local-Only Mode active (using rule-based feedback).")

# Whisper Model Size Selection
whisper_model_size = st.sidebar.selectbox(
    "Whisper Model Size",
    options=["tiny", "base", "small"],
    index=1,
    help="Base is recommended for CPU. Tiny is faster but less accurate. Small is highly accurate but slower."
)

# HuggingFace Model Selection
hf_model_name = st.sidebar.selectbox(
    "HuggingFace Suggestion Model",
    options=["google/bert_uncased_L-2_H-128_A-2", "prajjwal1/bert-tiny"],
    index=0,
    help="Tiny BERT models used for grammar context suggestions. Very lightweight (~17MB) and runs instantly."
)

# Confidence Threshold Slider
confidence_threshold = st.sidebar.slider(
    "Low-Confidence Threshold",
    min_value=0.50,
    max_value=0.90,
    value=0.70,
    step=0.05,
    help="Words with confidence scores below this threshold are flagged for review."
)

st.sidebar.markdown("---")
st.sidebar.header("🛡️ Privacy & Compliance (DPDP)")

# Consent Checkbox
dpdp_consent = st.sidebar.checkbox(
    "I consent to temporary processing of my voice recording",
    value=False,
    help="In compliance with DPDP, we require your consent. Audio files are processed in-memory or saved temporarily and deleted immediately after analysis."
)

st.sidebar.caption(
    "🔒 **Data Lifecycle Policy:** Uploaded and recorded audio files are never stored "
    "permanently. They are deleted immediately upon completion or failure of analysis."
)

# ----------------- MAIN CORE PROCESSOR -----------------

# Interactive Audio Tabs
tab_upload, tab_record = st.tabs(["📤 Upload Audio File", "🎤 Record Live"])

audio_source = None
uploaded_file = None
recorded_bytes = None

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload an English audio file (WAV, MP3, M4A format)",
        type=["wav", "mp3", "m4a"],
        help="Select a file that is between 30 and 45 seconds long."
    )
    if uploaded_file is not None:
        audio_source = "upload"

with tab_record:
    st.write("Click the mic icon to start recording. Speak for 30 to 45 seconds, then click it again to stop.")
    recorded_bytes = audio_recorder(
        text="Click to record",
        recording_color="#e74c3c",
        neutral_color="#34495e",
        icon_size="2x"
    )
    if recorded_bytes is not None:
        audio_source = "record"
        st.success("🎤 Audio successfully recorded!")

# Audio Processing and Analysis
if audio_source is not None:
    # 1. Audio Player UI component
    st.subheader("🔊 Audio Player")
    if audio_source == "upload":
        st.audio(uploaded_file, format=uploaded_file.type)
    else:
        st.audio(recorded_bytes, format="audio/wav")

    # 2. Analysis Trigger Section
    if not dpdp_consent:
        st.warning("⚠️ Please review and accept the DPDP data processing consent in the sidebar to enable analysis.")
        st.button("Analyze Pronunciation", disabled=True)
    else:
        if st.button("Analyze Pronunciation", type="primary"):
            # Setup temporary paths
            unique_id = str(uuid.uuid4())
            temp_input_path = None
            temp_converted_path = None

            try:
                # Progress container
                status_box = st.empty()
                progress_bar = st.progress(0.0)

                # Save the source to disk temporarily
                status_box.info("📁 Saving audio file temporarily...")
                progress_bar.progress(0.1)

                if audio_source == "upload":
                    ext = os.path.splitext(uploaded_file.name)[1]
                    temp_input_path = f"uploads/{unique_id}{ext}"
                    with open(temp_input_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                else:
                    temp_input_path = f"uploads/{unique_id}.wav"
                    with open(temp_input_path, "wb") as f:
                        f.write(recorded_bytes)

                # Validate audio duration (ffprobe check)
                status_box.info("⏳ Validating audio duration...")
                progress_bar.progress(0.2)
                
                try:
                    duration = validate_audio_duration(temp_input_path)
                    st.toast(f"Audio verified: {duration:.1f}s", icon="✅")
                except ValueError as e:
                    st.error(str(e))
                    st.stop()

                # Convert audio to standard mono 16kHz WAV
                status_box.info("⚙️ Converting audio format (FFmpeg 16kHz Mono)...")
                progress_bar.progress(0.3)
                temp_converted_path = f"uploads/{unique_id}_16k.wav"
                convert_to_wav(temp_input_path, temp_converted_path)

                # Load Whisper Model
                status_box.info(f"🧠 Loading Whisper model ({whisper_model_size}). Please wait...")
                progress_bar.progress(0.4)
                whisper_model = load_whisper_model(whisper_model_size)

                # Transcribe
                status_box.info("🗣️ Transcribing and extracting word timestamps...")
                progress_bar.progress(0.6)
                words, segments, info = transcribe_audio(whisper_model, temp_converted_path)

                if not words:
                    st.error("❌ No speech could be transcribed from the audio. Please speak clearly into the microphone.")
                    st.stop()

                # Score Heuristics
                status_box.info("📊 Evaluating pronunciation metrics...")
                progress_bar.progress(0.7)
                scoring_results = calculate_pronunciation_score(
                    words, 
                    segments, 
                    low_confidence_threshold=confidence_threshold,
                    audio_duration_seconds=duration
                )

                # HuggingFace & OpenAI Advice Integration
                status_box.info("🤖 Generating AI articulatory coaching suggestions...")
                progress_bar.progress(0.85)
                
                # Retrieve OpenAI/fallback advice
                flagged_words = scoring_results["flagged_words"]
                advice_list = generate_advice(flagged_words, openai_api_key)

                # Retrieve HuggingFace context predictions
                hf_pipeline = load_hf_pipeline(hf_model_name)
                
                # Combine feedback metrics
                combined_feedback = []
                for w_flagged in flagged_words:
                    word_str = w_flagged["word"]
                    conf = w_flagged["confidence"]
                    
                    # Context recommendations
                    hf_suggestions = get_hf_context_suggestions(hf_pipeline, words, w_flagged)
                    hf_str = ", ".join(hf_suggestions) if hf_suggestions else "N/A"
                    
                    # Look up articulatory advice
                    item_advice = next(
                        (item for item in advice_list if item["word"].lower().strip(".,?!") == word_str.lower().strip(".,?!")), 
                        None
                    )
                    
                    if item_advice:
                        issue = item_advice["issue"]
                        improvement = item_advice["suggested_improvement"]
                    else:
                        # Fallback if lookup failed
                        from utils.feedback import get_local_fallback_advice
                        fallback_item = get_local_fallback_advice(word_str)
                        issue = fallback_item["issue"]
                        improvement = fallback_item["suggested_improvement"]

                    combined_feedback.append({
                        "Word": word_str,
                        "Confidence": f"{conf * 100:.0f}%",
                        "Grammatical Context Alternatives (HF)": hf_str,
                        "Coaching Issue": issue,
                        "Pronunciation Tip": improvement
                    })

                progress_bar.progress(1.0)
                status_box.empty()
                st.success("✅ Analysis Complete!")

                # ----------------- DISPLAY RESULTS -----------------
                
                # Metric Columns
                st.markdown("## 📊 Evaluation Report")
                col_score, col_flagged, col_fillers, col_segments = st.columns(4)
                
                score_val = scoring_results["score"]
                
                # Coloring score based on quality
                if score_val >= 80:
                    score_color = "🟢"
                elif score_val >= 60:
                    score_color = "🟡"
                else:
                    score_color = "🔴"
                
                col_score.metric("Pronunciation Score", f"{score_val}/100", help="Starts at 100. Deducts points for unclear words, filler hesitations, and noisy segments.")
                col_flagged.metric("Flagged Mistakes", len(flagged_words))
                col_fillers.metric("Hesitations (Um/Uh)", len(scoring_results["hesitation_words"]))
                col_segments.metric("Unclear Segments", len(scoring_results["unclear_segments"]))

                # Enhanced Subscores UI
                st.markdown("### 🎓 ESL Diagnostic Subscores")
                col_acc, col_flu, col_cla, col_wpm = st.columns(4)
                
                col_acc.metric("Accuracy Subscore", f"{scoring_results['accuracy_score']}/100", help="Evaluates how clearly you pronounced individual content words.")
                col_flu.metric("Fluency Subscore", f"{scoring_results['fluency_score']}/100", help="Evaluates your pacing (WPM) and filler word hesitations.")
                col_cla.metric("Acoustic Clarity", f"{scoring_results['clarity_score']}/100", help="Evaluates overall audio recording noise and mumbled segments.")
                col_wpm.metric("Speaking Speed", f"{scoring_results['wpm']} WPM", delta=scoring_results['wpm_status'], delta_color="off", help="Standard speed ranges between 110 and 150 words per minute.")


                # Score Explanations
                with st.expander("Score Breakdown"):
                    st.write(f"**Base Score:** 100")
                    st.write(f"🔴 **Low-confidence words:** -{scoring_results['low_conf_deduction']} points (-3 per word)")
                    st.write(f"🟠 **Hesitation/Filler words:** -{scoring_results['hesitation_deduction']} points (-2 per filler)")
                    st.write(f"⚪ **Unclear segments/Noise:** -{scoring_results['unclear_segment_deduction']} points (-2 per segment)")
                    st.write(f"**Final Clamped Score:** {score_val}/100")

                # Display Transcript
                st.markdown("### 📝 Highlighted Transcript")
                st.caption("Hover over highlighted words to see their Whisper confidence score.")
                transcript_html = render_highlighted_transcript(words, confidence_threshold)
                st.markdown(transcript_html, unsafe_allow_html=True)

                # Mistakes & AI Advice Table
                st.markdown("### ⚠️ Flagged Pronunciation Issues")
                if len(combined_feedback) > 0:
                    df = pd.DataFrame(combined_feedback)
                    st.dataframe(
                        df, 
                        use_container_width=True,
                        column_config={
                            "Word": st.column_config.TextColumn(width="medium"),
                            "Confidence": st.column_config.TextColumn(width="small"),
                            "Grammatical Context Alternatives (HF)": st.column_config.TextColumn(width="medium"),
                            "Coaching Issue": st.column_config.TextColumn(width="large"),
                            "Pronunciation Tip": st.column_config.TextColumn(width="large")
                        }
                    )
                else:
                    st.success("🎉 Excellent! No words were flagged below the low-confidence threshold.")

                # Fluency Diagnostic Section (Fillers & Unclear segments)
                st.markdown("### 🗣️ Fluency & Audio Clarity Diagnostics")
                col_diag_fillers, col_diag_seg = st.columns(2)
                
                with col_diag_fillers:
                    st.markdown("**Hesitations/Filler Words Used**")
                    if scoring_results["hesitation_words"]:
                        fillers_found = [w["word"] for w in scoring_results["hesitation_words"]]
                        st.warning(f"Detected fillers: {', '.join(fillers_found)}")
                        st.caption("Reducing verbal pauses like 'um', 'uh', and 'ah' will make your speech sound more fluent and natural.")
                    else:
                        st.success("🟢 No verbal hesitations detected. Speech flow is smooth.")

                with col_diag_seg:
                    st.markdown("**Acoustic & Clarity Issues**")
                    if scoring_results["unclear_segments"]:
                        for idx, s in enumerate(scoring_results["unclear_segments"]):
                            st.info(f"Segment {idx+1} ({s['start']:.1f}s - {s['end']:.1f}s): *\"{s['text']}\"*")
                            st.caption(f"Reason: {s['reason']}")
                        st.caption("Segments are flagged if they contain excessive noise, mumbling, or average word confidence is very low.")
                    else:
                        st.success("🟢 Audio recording is clean with high acoustic clarity.")

            except Exception as e:
                st.error(f"An unexpected error occurred during processing: {str(e)}")
            
            finally:
                # CRITICAL: Clean up temporary files on disk to guarantee zero retention
                cleanup_file(temp_input_path)
                cleanup_file(temp_converted_path)

# backend/app.py
import os

# from dotenv import load_dotenv  # Import dotenv
# load_dotenv()  # Load .env file variables into os.environ BEFORE importing config
import uuid
import torch
import numpy as np
import torchaudio
import sqlite3
import base64
from scipy.spatial.distance import cosine, cdist
from scipy.stats import trim_mean
from datetime import datetime
import subprocess
import shutil
from flask import Flask, jsonify, request, render_template
from whitenoise import WhiteNoise
import logging
import json
import io
import config
import librosa
import webrtcvad

# --- Basic Logging Setup (uses config.LOG_LEVEL) ---
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),  # Set level from config
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
)

# --- Use Configuration from config.py ---
MODEL_NAME = config.MODEL_NAME
TARGET_SR = config.TARGET_SR

DATABASE_PATH = config.DATABASE_PATH
if DATABASE_PATH is None:
    logging.critical(
        "DATABASE_PATH is None after config import. This usually means directory creation failed. Exiting."
    )
    import sys

    sys.exit(1)
logging.info(f"Application will use database: {DATABASE_PATH}")
DB_NAME = os.path.basename(DATABASE_PATH)

MIN_REQUIRED_ENROLLMENT_SAMPLES = config.MIN_REQUIRED_ENROLLMENT_SAMPLES
ENROLLMENT_AUDIO_DURATION_S = config.ENROLLMENT_AUDIO_DURATION_S
PRIMARY_AGGREGATION_METHOD = config.PRIMARY_AGGREGATION_METHOD
# AGGREGATION_METHODS_TO_STORE_AND_BENCHMARK is directly from config now

ENABLE_ADAPTIVE_ENROLLMENT = config.ENABLE_ADAPTIVE_ENROLLMENT
ADAPTIVE_ENROLLMENT_CONFIDENCE_THRESHOLD = (
    config.ADAPTIVE_ENROLLMENT_CONFIDENCE_THRESHOLD
)
MAX_RAW_EMBEDDINGS_PER_USER = config.MAX_RAW_EMBEDDINGS_PER_USER
AUTHENTICATION_THRESHOLD = config.AUTHENTICATION_THRESHOLD


app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = config.FLASK_SECRET_KEY  # Use secret key from config
static_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.wsgi_app = WhiteNoise(app.wsgi_app, root=static_dir_path, prefix="/static/")

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None
if not FFMPEG_AVAILABLE:
    logging.critical("FFMPEG NOT FOUND!")
else:
    logging.info("FFMPEG found.")

speaker_encoder = None
try:
    logging.info(
        f"Initializing SpeechBrain EncoderClassifier from source: {MODEL_NAME}"
    )
    from speechbrain.inference.speaker import EncoderClassifier

    speaker_encoder = EncoderClassifier.from_hparams(
        source=MODEL_NAME, run_opts={"device": "cpu"}
    )
    logging.info("SpeechBrain EncoderClassifier initialized successfully.")
except Exception as e_init_sb:
    logging.critical(
        f"CRITICAL ERROR initializing SpeechBrain: {e_init_sb}", exc_info=True
    )


def init_db():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS speaker_profiles (
        voice_id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE,
        primary_aggregate_method TEXT NOT NULL, 
        primary_aggregated_voiceprint BLOB NOT NULL, 
        enrollment_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        last_seen_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP )"""
    )
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS raw_enrollment_embeddings (
        embedding_id INTEGER PRIMARY KEY AUTOINCREMENT, speaker_voice_id INTEGER NOT NULL,
        raw_embedding BLOB NOT NULL, quality_metrics_json TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (speaker_voice_id) REFERENCES speaker_profiles (voice_id) ON DELETE CASCADE )"""
    )
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS user_calculated_aggregates ( 
        stored_agg_id INTEGER PRIMARY KEY AUTOINCREMENT, speaker_voice_id INTEGER NOT NULL,
        aggregation_method_name TEXT NOT NULL, calculated_aggregate_embedding BLOB NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (speaker_voice_id) REFERENCES speaker_profiles (voice_id) ON DELETE CASCADE,
        UNIQUE (speaker_voice_id, aggregation_method_name) )"""
    )
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS authentication_benchmark_log (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT, speaker_voice_id INTEGER,            
        claimed_name TEXT NOT NULL, authentication_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        live_audio_quality_passed BOOLEAN, live_embedding_stored BOOLEAN DEFAULT FALSE,
        primary_auth_method TEXT, primary_auth_similarity REAL, primary_auth_decision BOOLEAN,         
        benchmark_scores_json TEXT, profile_was_adapted BOOLEAN DEFAULT FALSE,
        FOREIGN KEY (speaker_voice_id) REFERENCES speaker_profiles (voice_id) ON DELETE SET NULL )"""
    )
    conn.commit()
    conn.close()
    logging.info(f"DB '{DATABASE_PATH}' checked/initialized with all tables.")


init_db()


def serialize_embedding(arr):
    buffer = io.BytesIO()
    np.save(buffer, arr, allow_pickle=False)
    return buffer.getvalue()


def deserialize_embedding(b):
    buffer = io.BytesIO(b)
    return np.load(buffer, allow_pickle=False)


def convert_audio_to_wav(inp, outp):
    if not FFMPEG_AVAILABLE:
        if inp.lower().endswith(".wav"):
            try:
                shutil.copy(inp, outp)
                torchaudio.load(outp)
                logging.warning(f"FFmpeg N/A. Copied WAV: {inp} to {outp}")
                return True
            except Exception as e:
                logging.error(f"FFmpeg N/A. Copied WAV {inp} not loadable: {e}")
                return False
        logging.critical(f"FFmpeg N/A. Cannot convert {inp}.")
        return False
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        inp,
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(TARGET_SR),
        "-ac",
        "1",
        "-y",
        outp,
    ]
    try:
        logging.info(f"Converting: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=30)
        if os.path.exists(outp) and os.path.getsize(outp) > 0:
            logging.info(f"Converted {inp} to {outp}")
            return True
        logging.error(f"FFmpeg empty output for {inp}.")
        return False
    except Exception as e:
        logging.error(f"FFmpeg err {inp}: {e}", exc_info=True)
        return False


# SNR Calculation
def calculate_snr_vad(audio_np, sr, frame_duration_ms=30, vad_aggressiveness=1):
    """
    Estimates SNR using Voice Activity Detection.
    Returns SNR in dB, or None if calculation fails.
    """
    if sr not in [8000, 16000, 32000, 48000]:
        logging.warning(
            f"SNR VAD: Unsupported sample rate {sr} for webrtcvad. Needs 8k, 16k, 32k, or 48k."
        )
        # Optionally, try to resample here if you want to force it,
        # but your TARGET_SR should align with one of these.
        return None
    if audio_np.dtype != np.int16:  # webrtcvad expects 16-bit int samples
        # Convert float audio (-1 to 1) to int16
        if np.max(np.abs(audio_np)) <= 1.0:
            audio_int16 = (audio_np * 32767).astype(np.int16)
        else:  # Assuming it's already in a different int range, less ideal
            logging.warning(
                "SNR VAD: Audio not in float (-1 to 1) or int16. SNR might be inaccurate."
            )
            audio_int16 = audio_np.astype(np.int16)  # Attempt conversion
    else:
        audio_int16 = audio_np

    vad = webrtcvad.Vad()
    vad.set_mode(vad_aggressiveness)  # 0 (least aggressive) to 3 (most aggressive)

    samples_per_frame = int(sr * frame_duration_ms / 1000)

    speech_frames_power = []
    noise_frames_power = []

    num_frames = len(audio_int16) // samples_per_frame
    if num_frames == 0:
        logging.warning("SNR VAD: Audio too short for any frames.")
        return None

    for i in range(num_frames):
        start_byte = i * samples_per_frame
        end_byte = start_byte + samples_per_frame
        frame = audio_int16[start_byte:end_byte]

        # Ensure the frame is bytes for webrtcvad
        try:
            is_speech = vad.is_speech(frame.tobytes(), sr)
        except Exception as e:
            # This can happen if frame length is wrong for sr for webrtcvad
            logging.warning(f"SNR VAD: Error during VAD.is_speech for frame {i}: {e}")
            continue  # Skip problematic frame

        # Calculate power of the original float frame (for better precision)
        original_float_frame = audio_np[start_byte:end_byte]
        power = np.sum(original_float_frame**2) / len(original_float_frame)

        if is_speech:
            speech_frames_power.append(power)
        else:
            noise_frames_power.append(power)

    if not speech_frames_power:
        logging.warning("SNR VAD: No speech frames detected.")
        return -np.inf  # Or some other indicator of no speech
    if not noise_frames_power:
        logging.warning(
            "SNR VAD: No noise frames detected (or all frames are speech). Assuming very low noise for SNR calc."
        )
        # Assign a very small epsilon for noise power to avoid division by zero
        # and to represent a high SNR scenario.
        # This choice can impact the result when there's genuinely no noise.
        avg_noise_power = 1e-10  # A very small non-zero power
    else:
        avg_noise_power = np.mean(noise_frames_power)

    avg_speech_power = np.mean(speech_frames_power)

    if avg_noise_power == 0:  # Should be handled by epsilon above, but as a safeguard
        if avg_speech_power > 0:
            return np.inf  # Essentially infinite SNR if noise is truly zero
        else:
            return 0  # Or None, if speech power is also zero

    snr_val = 10 * np.log10(avg_speech_power / avg_noise_power)
    return snr_val


def assess_audio_quality(
    arr,
    sr,
    min_speech=1.0,
    clip_perc=1.5,
    var_thresh=1e-6,
    min_snr_db=config.MIN_SNR_DB_THRESHOLD,
):  # Add min_snr_db
    qr = {"is_usable": True, "reasons": [], "metrics": {}}
    dur = len(arr) / sr if sr > 0 else 0
    qr["metrics"]["dur"] = round(dur, 2)

    if sr != TARGET_SR:  # TARGET_SR should be defined globally from config
        qr["is_usable"] = False
        qr["reasons"].append(f"SR ERR {sr}(exp {TARGET_SR})")

    sd = dur  # Assuming speech duration for now, VAD would make this more accurate
    if sd < min_speech:
        qr["is_usable"] = False
        qr["reasons"].append(
            f"Short audio ({sd:.1f}s<{min_speech}s)"
        )  # Changed from "speech" to "audio"

    ma = np.max(np.abs(arr)) if len(arr) > 0 else 0
    na = arr / ma if ma > 0 else arr  # Use arr directly if ma is 0 (silence)
    clip_threshold = 0.985
    if ma > 0:  # Only calculate clipping if there's some signal
        percentage_near_peak = (
            (np.sum(np.abs(arr) >= clip_threshold * ma) / len(arr)) * 100
            if len(arr) > 0
            else 0
        )
    else:  # if 'ma' is 0 (silence), clipping is 0
        percentage_near_peak = 0.0

    qr["metrics"]["clipping_percentage"] = round(percentage_near_peak, 1)
    if percentage_near_peak > clip_perc:
        qr["is_usable"] = False
        qr["reasons"].append(f"Clip({percentage_near_peak:.1f}%>{clip_perc}%)")

    v = np.var(arr) if len(arr) > 0 else 0
    qr["metrics"]["variance"] = float(f"{v:.1e}")
    if v < var_thresh and sd > 0.1:  # only for non-trivial duration
        qr["is_usable"] = False
        qr["reasons"].append(f"Low var({v:.1e}<{var_thresh})")

    # --- SNR Calculation ---
    snr_db = None
    if sr == TARGET_SR and len(arr) > 0:  # Calculate SNR if basic conditions met
        # `calculate_snr_vad` expects float audio `arr` and sample rate `sr`
        # It will handle conversion to int16 internally if `arr` is float.
        snr_db = calculate_snr_vad(
            arr, sr, vad_aggressiveness=2
        )  # vad_aggressiveness 0-3

    if snr_db is not None:
        qr["metrics"]["snr_db"] = round(snr_db, 1)
        if snr_db < min_snr_db:
            qr["is_usable"] = False
            qr["reasons"].append(f"Low SNR ({snr_db:.1f}dB<{min_snr_db}dB)")
    else:
        qr["metrics"]["snr_db"] = "N/A"
        # Optionally, consider it unusable if SNR can't be calculated and it's important
        # qr["is_usable"] = False
        # qr["reasons"].append("SNR Calc Failed")

    if not qr["reasons"] and qr["is_usable"]:  # If no other reasons, and still usable
        qr["reasons"].append("OK")
    elif (
        not qr["reasons"] and not qr["is_usable"]
    ):  # If set to not usable by some logic but no reason recorded
        qr["reasons"].append("Quality Check Failed (Unknown)")

    logging.info(
        f"Quality:Usable={qr['is_usable']}, R={qr['reasons']}, Metrics={qr['metrics']}"
    )
    return qr


# --- Define ALL possible aggregate functions ---
def _aggregate_mean(embs):
    return np.mean(np.stack(embs), axis=0) if embs and len(embs) > 0 else None


def _aggregate_median(embs):
    return np.median(np.stack(embs), axis=0) if embs and len(embs) > 0 else None


def _aggregate_medoid(embs, metric="cosine"):
    if not embs or len(embs) == 0:
        return None
    if len(embs) == 1:
        return embs[0]
    try:
        mat = np.vstack(embs)
        dm = cdist(mat, mat, metric=metric)
        return embs[np.argmin(np.sum(dm, axis=1))]
    except ValueError:
        logging.warning("Medoid issue,ret first.")
        return embs[0] if embs else None


def _aggregate_trimmed_mean(embeddings, prop=0.1):
    if not embeddings or len(embeddings) == 0:
        return None
    min_s = np.ceil(1 / (1 - 2 * prop)) if prop < 0.5 and prop > 0 else 1
    if len(embeddings) >= min_s and prop < 0.5:
        return trim_mean(np.stack(embeddings), proportiontocut=prop, axis=0)
    return _aggregate_mean(embeddings)  # Fallback


def _aggregate_max_pool(embs):
    return np.max(np.stack(embs), axis=0) if embs and len(embs) > 0 else None


def _aggregate_min_pool(embs):
    return np.min(np.stack(embs), axis=0) if embs and len(embs) > 0 else None


# --- Populate AGGREGATION_FUNCTIONS_FOR_BENCHMARK based on config ---
ALL_POSSIBLE_AGGREGATION_FUNCTIONS = {  # Master dictionary
    "mean": _aggregate_mean,
    "median": _aggregate_median,
    "medoid": _aggregate_medoid,
    "trimmed_mean": _aggregate_trimmed_mean,
    "max_pool": _aggregate_max_pool,
    "min_pool": _aggregate_min_pool,
}

AGGREGATION_FUNCTIONS_FOR_BENCHMARK = {
    name: func
    for name, func in ALL_POSSIBLE_AGGREGATION_FUNCTIONS.items()
    if name in config.AGGREGATION_METHODS_TO_USE_AND_BENCHMARK  # Filter based on config
}

# Ensure PRIMARY_AGGREGATION_METHOD is valid and present in the benchmarkable functions
if PRIMARY_AGGREGATION_METHOD not in AGGREGATION_FUNCTIONS_FOR_BENCHMARK:
    logging.warning(
        f"Primary method {PRIMARY_AGGREGATION_METHOD} is not in the configured AGGREGATION_METHODS_TO_USE_AND_BENCHMARK ({config.AGGREGATION_METHODS_TO_USE_AND_BENCHMARK}). "
        f"Defaulting to '{config.DEFAULT_PRIMARY_AGGREGATION_METHOD}' or first available if that's also not present."
    )
    if config.DEFAULT_PRIMARY_AGGREGATION_METHOD in AGGREGATION_FUNCTIONS_FOR_BENCHMARK:
        PRIMARY_AGGREGATION_METHOD = config.DEFAULT_PRIMARY_AGGREGATION_METHOD
    elif AGGREGATION_FUNCTIONS_FOR_BENCHMARK:
        PRIMARY_AGGREGATION_METHOD = next(iter(AGGREGATION_FUNCTIONS_FOR_BENCHMARK))
    else:  # Should not happen if config.AGGREGATION_METHODS_TO_USE_AND_BENCHMARK is not empty
        logging.critical(
            "CRITICAL: No valid aggregation methods configured for benchmark or primary use. App will likely fail."
        )
        PRIMARY_AGGREGATION_METHOD = "mean"  # A desperate default

logging.info(f"Effective Primary Aggregation Method: {PRIMARY_AGGREGATION_METHOD}")
logging.info(
    f"Benchmark will use methods: {list(AGGREGATION_FUNCTIONS_FOR_BENCHMARK.keys())}"
)


def get_single_embedding_and_quality(audio_b64_str, temp_file_prefix="s_"):
    logging.info(
        f"get_single_embedding_and_quality called for prefix: {temp_file_prefix}"
    )
    if speaker_encoder is None:
        return (
            None,
            None,
            {"is_usable": False, "reasons": ["Encoder N/A"], "metrics": {}},
        )
    uuid_s = str(uuid.uuid4())
    tmp_dir = os.path.join(os.path.dirname(__file__), "temp_audio_files")
    os.makedirs(tmp_dir, exist_ok=True)
    orig_tmp_pth, proc_wav_pth = None, None
    emb_np, raw_bytes_for_db, qr = None, None, {}
    try:
        logging.info(f"Attempting to decode base64 for {temp_file_prefix}")
        if "," not in audio_b64_str:
            raise ValueError("Invalid base64 URI.")
        hdr, enc_data = audio_b64_str.split(",", 1)
        raw_bytes_for_db = base64.b64decode(enc_data)
        mime = (
            hdr.split(";")[0].split(":")[1]
            if hdr.startswith("data:") and ";" in hdr
            else "app/octet"
        )
        ext = {
            "audio/wav": ".wav",
            "audio/webm": ".webm",
            "audio/ogg": ".ogg",
            "audio/mpeg": ".mp3",
            "audio/mp4": ".m4a",
            "audio/aac": ".aac",
        }.get(mime, ".bin")
        orig_tmp_pth = os.path.join(tmp_dir, f"in_{temp_file_prefix}{uuid_s}{ext}")
        proc_wav_pth = os.path.join(tmp_dir, f"proc_{temp_file_prefix}{uuid_s}.wav")
        with open(orig_tmp_pth, "wb") as f:
            f.write(raw_bytes_for_db)
        if not convert_audio_to_wav(orig_tmp_pth, proc_wav_pth):
            return (
                None,
                raw_bytes_for_db,
                {"is_usable": False, "reasons": ["FFmpeg fail"], "metrics": {}},
            )
        sig, sr_orig = torchaudio.load(proc_wav_pth)
        if sig.ndim > 1 and sig.shape[0] > 1:
            sig = sig[0, :].unsqueeze(0)
        elif sig.ndim == 1:
            sig = sig.unsqueeze(0)
        if sr_orig != TARGET_SR:
            sig = torchaudio.transforms.Resample(orig_freq=sr_orig, new_freq=TARGET_SR)(
                sig
            )
        proc_np = sig.squeeze().cpu().numpy()
        qr = assess_audio_quality(proc_np, TARGET_SR)
        if qr.get("is_usable", False):
            emb_tensor = speaker_encoder.encode_batch(sig)
            emb_np = emb_tensor.squeeze().detach().cpu().numpy()
        return emb_np, raw_bytes_for_db, qr
    except Exception as e:
        logging.error(f"Err get_single_emb: {e}", exc_info=True)
        return (
            None,
            raw_bytes_for_db,
            {
                "is_usable": False,
                "reasons": [f"Proc err: {str(e)[:50]}"],
                "metrics": {},
            },
        )
    finally:
        for p in [orig_tmp_pth, proc_wav_pth]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception as e_d:
                    logging.warning(f"Failed del temp {p}: {e_d}")


def log_benchmark_attempt(
    cursor,
    speaker_id,
    claimed_name,
    live_quality_passed,
    live_emb_stored,
    primary_method,
    primary_sim,
    primary_decision,
    benchmark_scores_dict,
    adapted,
):
    # ... (Same robust version as before) ...
    try:
        benchmark_json = (
            json.dumps(benchmark_scores_dict) if benchmark_scores_dict else None
        )
        cursor.execute(
            """INSERT INTO authentication_benchmark_log (speaker_voice_id, claimed_name, live_audio_quality_passed, live_embedding_stored, primary_auth_method, primary_auth_similarity, primary_auth_decision, benchmark_scores_json, profile_was_adapted) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                speaker_id,
                claimed_name,
                bool(live_quality_passed),
                bool(live_emb_stored),
                primary_method,
                float(primary_sim) if primary_sim is not None else None,
                bool(primary_decision) if primary_decision is not None else None,
                benchmark_json,
                bool(adapted),
            ),
        )
        logging.info(
            f"Logged benchmark attempt for '{claimed_name}' (User ID: {speaker_id})"
        )
    except Exception as e_log:
        logging.error(
            f"Failed to log benchmark for '{claimed_name}': {e_log}", exc_info=True
        )


@app.route("/")
def index_page():
    return render_template(
        "index.html",
        MIN_REQUIRED_ENROLLMENT_SAMPLES=config.MIN_REQUIRED_ENROLLMENT_SAMPLES,
        ENROLLMENT_AUDIO_DURATION_S=config.ENROLLMENT_AUDIO_DURATION_S,
        MIN_SNR_DB_THRESHOLD_CONFIG=config.MIN_SNR_DB_THRESHOLD,
    )


@app.route("/register", methods=["POST"])
def register():
    if not request.is_json:
        return jsonify({"s": False, "e": "JSON expected."}), 400
    data = request.get_json()
    name = data.get("name")
    audios_b64 = data.get("audios")
    if not name:
        return jsonify({"s": False, "e": "Name missing."}), 400
    if (
        not audios_b64
        or not isinstance(audios_b64, list)
        or len(audios_b64) < MIN_REQUIRED_ENROLLMENT_SAMPLES
    ):
        return (
            jsonify(
                {
                    "s": False,
                    "e": f"Need {MIN_REQUIRED_ENROLLMENT_SAMPLES} samples (got {len(audios_b64) if audios_b64 else 0}).",
                }
            ),
            400,
        )
    logging.info(
        f"Reg for '{name}' with {len(audios_b64)} samples. Primary method: '{PRIMARY_AGGREGATION_METHOD}'."
    )

    q_reports_all_samples = []
    valid_sample_details_for_db = []
    valid_embs_np = []

    # <<< START OF LOOP A: Process ALL submitted audio samples >>>
    for i, ab64 in enumerate(audios_b64):
        logging.info(f"Processing sample {i+1}/{len(audios_b64)}...")
        emb_np, _, qr_original = get_single_embedding_and_quality(
            ab64, temp_file_prefix=f"enr_{i}_"
        )
        q_reports_all_samples.append(qr_original)  # Collect ALL original reports

        if emb_np is not None and qr_original.get("is_usable", False):
            valid_embs_np.append(emb_np)
            # ... (your sanitization logic for valid_sample_details_for_db as before) ...
            original_metrics = qr_original.get("metrics", {})
            sanitized_metrics_for_db = {}
            for key, value in original_metrics.items():
                if isinstance(value, (np.float32, np.float64)):
                    sanitized_metrics_for_db[key] = float(value)
                elif isinstance(value, (np.int32, np.int64, np.int_)):
                    sanitized_metrics_for_db[key] = int(value)
                elif isinstance(value, np.bool_):
                    sanitized_metrics_for_db[key] = bool(value)
                elif isinstance(value, (float, int, str, bool)):
                    sanitized_metrics_for_db[key] = value
                else:
                    sanitized_metrics_for_db[key] = str(value)
            valid_sample_details_for_db.append(
                {"embedding": emb_np, "quality_metrics": sanitized_metrics_for_db}
            )
        else:
            logging.warning(
                f"Sample {i+1} for '{name}' failed quality. R: {qr_original.get('reasons')}"
            )
    # <<< END OF LOOP A >>>

    # --- This section now correctly happens AFTER ALL samples are processed ---
    logging.info(
        f"LOOP A FINISHED. Total raw quality reports collected (q_reports_all_samples): {len(q_reports_all_samples)}"
    )  # DEBUG 1

    detailed_quality_reports_for_response = []  # Initialize for API response
    for i_report, qr_rep_original_for_api in enumerate(
        q_reports_all_samples
    ):  # Iterate ALL collected reports
        metrics_rep = qr_rep_original_for_api.get("metrics", {})
        raw_snr_value_rep = metrics_rep.get("snr_db", "N/A")
        snr_value_serializable_rep = "N/A"

        if isinstance(raw_snr_value_rep, (np.float32, np.float64, float)):
            py_float_snr_rep = float(raw_snr_value_rep)
            if (
                py_float_snr_rep == float("inf")
                or py_float_snr_rep == -float("inf")
                or py_float_snr_rep != py_float_snr_rep
            ):
                snr_value_serializable_rep = str(py_float_snr_rep)
            else:
                snr_value_serializable_rep = py_float_snr_rep
        elif raw_snr_value_rep is not None:
            snr_value_serializable_rep = str(raw_snr_value_rep)

        report_for_api = {
            "sample_index": i_report + 1,
            "is_usable": qr_rep_original_for_api.get("is_usable", False),
            "reasons": qr_rep_original_for_api.get("reasons", ["Unknown"]),
            "snr_db": snr_value_serializable_rep,
        }
        detailed_quality_reports_for_response.append(report_for_api)

    logging.info(
        f"Number of reports processed for API response: {len(detailed_quality_reports_for_response)}"
    )  # DEBUG 2
    try:
        logging.info(
            f"API Response - detailed_quality_reports (first item if any): {json.dumps(detailed_quality_reports_for_response[0] if detailed_quality_reports_for_response else {})}"
        )
        logging.info(
            f"API Response - detailed_quality_reports (full): {json.dumps(detailed_quality_reports_for_response)}"
        )
    except Exception as e_json_debug_api:
        logging.error(
            f"Error JSON dumping debug info for API detailed_quality_reports: {e_json_debug_api}"
        )

    # --- NOW, check if enough good samples were collected AFTER processing ALL of them ---
    if len(valid_embs_np) < MIN_REQUIRED_ENROLLMENT_SAMPLES:
        logging.warning(
            f"FINAL CHECK: Not enough good samples: {len(valid_embs_np)} found, {MIN_REQUIRED_ENROLLMENT_SAMPLES} required."
        )  # DEBUG 4
        return (
            jsonify(
                {
                    "s": False,
                    "e": f"Need {MIN_REQUIRED_ENROLLMENT_SAMPLES} good quality samples, got {len(valid_embs_np)}.",
                    "sample_reports": detailed_quality_reports_for_response,  # Will now contain reports for ALL samples
                }
            ),
            400,
        )

    # --- If we reach here, we have enough good samples. Proceed with successful registration path. ---
    logging.info("Proceeding with successful registration path.")  # DEBUG INFO

    primary_agg_func_dispatch = AGGREGATION_FUNCTIONS_FOR_BENCHMARK.get(
        PRIMARY_AGGREGATION_METHOD
    )
    # ... (rest of your successful registration logic, using valid_embs_np and valid_sample_details_for_db)
    # ... ensure the success jsonify also sends detailed_quality_reports_for_response
    if not primary_agg_func_dispatch:  # etc.
        logging.error(
            f"Primary method {PRIMARY_AGGREGATION_METHOD} not in AGGREGATION_FUNCTIONS_FOR_BENCHMARK."
        )
        return (
            jsonify(
                {
                    "s": False,
                    "e": "Server agg config error.",
                    "sample_reports": detailed_quality_reports_for_response,
                }
            ),
            500,
        )

    primary_aggregated_emb = primary_agg_func_dispatch(valid_embs_np)
    if primary_aggregated_emb is None:
        return (
            jsonify(
                {
                    "s": False,
                    "e": f"Primary agg ({PRIMARY_AGGREGATION_METHOD}) failed.",
                    "sample_reports": detailed_quality_reports_for_response,
                }
            ),
            500,
        )

    conn = sqlite3.connect(DATABASE_PATH)
    cur = conn.cursor()
    try:
        # ... (your DB insert logic using `valid_sample_details_for_db` for quality_metrics_json) ...
        ser_prim_emb = serialize_embedding(primary_aggregated_emb)
        cur.execute(
            "INSERT INTO speaker_profiles(name,primary_aggregate_method,primary_aggregated_voiceprint) VALUES(?,?,?)",
            (name, PRIMARY_AGGREGATION_METHOD, ser_prim_emb),
        )
        spk_id = cur.lastrowid

        for detail_for_db in valid_sample_details_for_db:
            cur.execute(
                "INSERT INTO raw_enrollment_embeddings(speaker_voice_id,raw_embedding,quality_metrics_json) VALUES(?,?,?)",
                (
                    spk_id,
                    serialize_embedding(detail_for_db["embedding"]),
                    json.dumps(detail_for_db["quality_metrics"]),
                ),
            )
        for (
            method_name_iter,
            agg_func_iter,
        ) in AGGREGATION_FUNCTIONS_FOR_BENCHMARK.items():
            # ... (your logic for calculated aggregates) ...
            current_iter_agg_emb = None  # (rest of your aggregate logic)
            if method_name_iter == "medoid":
                current_iter_agg_emb = agg_func_iter(valid_embs_np, metric="cosine")
            else:
                current_iter_agg_emb = agg_func_iter(valid_embs_np)
            if current_iter_agg_emb is not None:
                cur.execute(
                    "INSERT INTO user_calculated_aggregates(speaker_voice_id,aggregation_method_name,calculated_aggregate_embedding) VALUES(?,?,?)",
                    (
                        spk_id,
                        method_name_iter,
                        serialize_embedding(current_iter_agg_emb),
                    ),
                )

        conn.commit()
        logging.info(
            "Successful registration. Preparing to send success response with sample reports."
        )
        return jsonify(  # SUCCESS RESPONSE
            {
                "s": True,
                "vId": spk_id,
                "name": name,
                "m": PRIMARY_AGGREGATION_METHOD,
                "n_s": len(valid_embs_np),
                "sample_reports": detailed_quality_reports_for_response,  # Crucial
            }
        )
    # ... (except blocks as before, ensuring they also pass detailed_quality_reports_for_response)
    except sqlite3.IntegrityError:
        logging.warning(
            f"Integrity error for name {name}. Sending 409 with full reports."
        )
        return (
            jsonify(
                {
                    "s": False,
                    "e": f"Name '{name}' exists.",
                    "sample_reports": detailed_quality_reports_for_response,
                }
            ),
            409,
        )
    except Exception as e:
        conn.rollback()
        logging.error(f"DB err reg:{e}", exc_info=True)
        return (
            jsonify(
                {
                    "s": False,
                    "e": "DB reg err.",
                    "sample_reports": detailed_quality_reports_for_response,
                }
            ),
            500,
        )
    finally:
        if conn:
            conn.close()


@app.route("/authenticate", methods=["POST"])
def authenticate():
    if not request.is_json:
        return jsonify({"success": False, "error": "JSON expected."}), 400
    data = request.get_json()
    claimed_name = data.get("name")
    audio_data_base64 = data.get("audio")
    if not claimed_name or not audio_data_base64:
        return jsonify({"success": False, "error": "Name and audio required."}), 400
    (
        live_embedding_np,
        live_raw_audio_bytes,
        quality_report,
    ) = get_single_embedding_and_quality(audio_data_base64, temp_file_prefix="auth_")
    voice_id_log = None
    primary_auth_method_log = None
    primary_sim_log = None
    primary_decision_log = None
    benchmark_log_data = {}
    profile_updated_adaptively = False
    live_embedding_stored_flag = False
    auth_message = "Authentication process initiated."
    if live_embedding_np is None or not quality_report.get("is_usable", False):
        auth_message = f"Live audio quality issue: {', '.join(quality_report.get('reasons',['Unknown']))}"
        conn_log_fail = sqlite3.connect(DATABASE_PATH)
        cursor_log_fail = conn_log_fail.cursor()
        try:
            log_benchmark_attempt(
                cursor_log_fail,
                None,
                claimed_name,
                False,
                False,
                None,
                None,
                None,
                {"error": auth_message, "details": quality_report.get("reasons")},
                False,
            )
            conn_log_fail.commit()
        except Exception as e_log:
            logging.error(f"Failed to log bad audio quality for {claimed_name}:{e_log}")
        finally:
            if conn_log_fail:
                conn_log_fail.close()
        return (
            jsonify(
                {"success": False, "authenticated": False, "message": auth_message}
            ),
            400,
        )
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT voice_id,name,primary_aggregate_method,primary_aggregated_voiceprint FROM speaker_profiles WHERE name=?",
            (claimed_name,),
        )
        target_user = cursor.fetchone()
        if not target_user:
            auth_message = f"User '{claimed_name}' not registered."
            log_benchmark_attempt(
                cursor,
                None,
                claimed_name,
                True,
                False,
                None,
                None,
                None,
                {"info": auth_message},
                False,
            )
            conn.commit()
            return jsonify(
                {"success": True, "authenticated": False, "message": auth_message}
            )
        (
            voice_id,
            speaker_name_db,
            stored_primary_method,
            stored_primary_voiceprint_bytes,
        ) = target_user
        voice_id_log = voice_id
        primary_auth_method_log = stored_primary_method
        stored_primary_agg_emb_np = deserialize_embedding(
            stored_primary_voiceprint_bytes
        )
        primary_live_emb_to_compare = live_embedding_np
        if primary_live_emb_to_compare.shape != stored_primary_agg_emb_np.shape:
            auth_message = (
                "Internal config error: Primary voiceprint dimension mismatch."
            )
            logging.error(
                f"Primary Auth Dim Mismatch for {speaker_name_db}! Live:{primary_live_emb_to_compare.shape},Stored:{stored_primary_agg_emb_np.shape}. Method:{stored_primary_method}"
            )
            log_benchmark_attempt(
                cursor,
                voice_id,
                claimed_name,
                True,
                False,
                stored_primary_method,
                None,
                None,
                {"error": auth_message},
                False,
            )
            conn.commit()
            return (
                jsonify(
                    {"success": False, "authenticated": False, "message": auth_message}
                ),
                500,
            )
        primary_similarity = 1 - cosine(
            primary_live_emb_to_compare, stored_primary_agg_emb_np
        )
        threshold = AUTHENTICATION_THRESHOLD  # Use config threshold
        primary_authenticated = bool(primary_similarity >= threshold)
        primary_sim_log = primary_similarity
        primary_decision_log = primary_authenticated
        logging.info(
            f"Primary Auth({stored_primary_method}) for '{speaker_name_db}':Sim={primary_similarity:.4f},Auth={primary_authenticated}"
        )
        if primary_authenticated:
            cursor.execute(
                "UPDATE speaker_profiles SET last_seen_timestamp=? WHERE voice_id=?",
                (datetime.now().isoformat(), voice_id),
            )
            if (
                ENABLE_ADAPTIVE_ENROLLMENT
                and primary_similarity >= ADAPTIVE_ENROLLMENT_CONFIDENCE_THRESHOLD
            ):
                logging.info(f"High confidence for {speaker_name_db}. Adaptive update.")
                try:
                    live_auth_metrics_original = quality_report.get("metrics", {})
                    sanitized_live_auth_metrics = {}
                    for key, value in live_auth_metrics_original.items():
                        if isinstance(value, (np.float32, np.float64)):
                            # Handle potential inf/nan from numpy floats before converting to Python float
                            if np.isinf(value) or np.isnan(value):
                                sanitized_live_auth_metrics[key] = str(value) # "inf", "-inf", "nan"
                            else:
                                sanitized_live_auth_metrics[key] = float(value)
                        elif isinstance(value, (np.int_, np.intc, np.intp, np.int8,
                                                np.int16, np.int32, np.int64, np.uint8,
                                                np.uint16, np.uint32, np.uint64)): # More comprehensive int check
                            sanitized_live_auth_metrics[key] = int(value)
                        elif isinstance(value, np.bool_):
                            sanitized_live_auth_metrics[key] = bool(value)
                        # Handle Python's special floats if they somehow occur here directly
                        elif isinstance(value, float) and (value == float('inf') or value == -float('inf') or value != value): # value != value checks for NaN
                             sanitized_live_auth_metrics[key] = str(value)
                        elif isinstance(value, (float, int, str, bool)): # Already python primitive and finite
                            sanitized_live_auth_metrics[key] = value
                        else: 
                            # For any other unhandled types, convert to string as a fallback
                            sanitized_live_auth_metrics[key] = str(value)
                    new_q_metrics_json = json.dumps(sanitized_live_auth_metrics)
                    
                    cursor.execute(
                        "INSERT INTO raw_enrollment_embeddings (speaker_voice_id,raw_embedding,quality_metrics_json,timestamp) VALUES (?,?,?,?)",
                        (
                            voice_id,
                            serialize_embedding(live_embedding_np),
                            new_q_metrics_json,
                            datetime.now().isoformat(),
                        ),
                    )
                    live_embedding_stored_flag = True
                    if MAX_RAW_EMBEDDINGS_PER_USER > 0:
                        cursor.execute(
                            "SELECT COUNT(*) FROM raw_enrollment_embeddings WHERE speaker_voice_id=?",
                            (voice_id,),
                        )
                        count = cursor.fetchone()[0]
                        if count > MAX_RAW_EMBEDDINGS_PER_USER:
                            num_del = count - MAX_RAW_EMBEDDINGS_PER_USER
                            cursor.execute(
                                "DELETE FROM raw_enrollment_embeddings WHERE embedding_id IN (SELECT embedding_id FROM raw_enrollment_embeddings WHERE speaker_voice_id=? ORDER BY timestamp ASC LIMIT ?)",
                                (voice_id, num_del),
                            )
                    cursor.execute(
                        "SELECT raw_embedding FROM raw_enrollment_embeddings WHERE speaker_voice_id=?",
                        (voice_id,),
                    )
                    updated_raw_embs = [
                        deserialize_embedding(r[0]) for r in cursor.fetchall()
                    ]
                    if updated_raw_embs:
                        primary_agg_func_reapply = (
                            AGGREGATION_FUNCTIONS_FOR_BENCHMARK.get(
                                stored_primary_method
                            )
                        )
                        if primary_agg_func_reapply:
                            new_primary_agg = primary_agg_func_reapply(updated_raw_embs)
                            if new_primary_agg is not None:
                                cursor.execute(
                                    "UPDATE speaker_profiles SET primary_aggregated_voiceprint=? WHERE voice_id=?",
                                    (serialize_embedding(new_primary_agg), voice_id),
                                )
                                profile_updated_adaptively = True
                        for (
                            meth_name_iter,
                            agg_f_iter,
                        ) in AGGREGATION_FUNCTIONS_FOR_BENCHMARK.items():
                            recalculated_agg = (
                                agg_f_iter(updated_raw_embs)
                                if meth_name_iter != "medoid"
                                else agg_f_iter(updated_raw_embs, metric="cosine")
                            )
                            if recalculated_agg is not None:
                                cursor.execute(
                                    "UPDATE user_calculated_aggregates SET calculated_aggregate_embedding=?,timestamp=? WHERE speaker_voice_id=? AND aggregation_method_name=?",
                                    (
                                        serialize_embedding(recalculated_agg),
                                        datetime.now().isoformat(),
                                        voice_id,
                                        meth_name_iter,
                                    ),
                                )
                                if cursor.rowcount == 0:
                                    cursor.execute(
                                        "INSERT INTO user_calculated_aggregates (speaker_voice_id,aggregation_method_name,calculated_aggregate_embedding,timestamp) VALUES (?,?,?,?)",
                                        (
                                            voice_id,
                                            meth_name_iter,
                                            serialize_embedding(recalculated_agg),
                                            datetime.now().isoformat(),
                                        ),
                                    )
                except Exception as e_adapt:
                    logging.error(
                        f"Adaptive update error {voice_id}:{e_adapt}", exc_info=True
                    )
            conn.commit()
        cursor.execute(
            "SELECT aggregation_method_name,calculated_aggregate_embedding FROM user_calculated_aggregates WHERE speaker_voice_id=?",
            (voice_id,),
        )
        stored_aggregates_for_benchmark = cursor.fetchall()
        if not stored_aggregates_for_benchmark:
            logging.warning(
                f"No stored aggregates for {speaker_name_db} for benchmark."
            )
        else:
            for (
                method_name_bench,
                stored_agg_emb_bytes_bench,
            ) in stored_aggregates_for_benchmark:
                agg_emb_np_bench = deserialize_embedding(stored_agg_emb_bytes_bench)
                if live_embedding_np.shape == agg_emb_np_bench.shape:
                    sim = 1 - cosine(live_embedding_np, agg_emb_np_bench)
                    benchmark_log_data[method_name_bench] = {
                        "similarity": round(float(sim), 4),
                        "matches_threshold": bool(sim >= threshold),
                    }
                else:
                    benchmark_log_data[method_name_bench] = {
                        "similarity": None,
                        "error": f"Bench Dim Mismatch {live_embedding_np.shape} vs {agg_emb_np_bench.shape}",
                    }
        auth_message = f"Primary auth ({stored_primary_method}) {'succeeded' if primary_authenticated else 'failed'}. Score: {primary_similarity:.4f}"
        if profile_updated_adaptively:
            auth_message += " [Profile Adapted]"
        log_benchmark_attempt(
            cursor,
            voice_id_log,
            claimed_name,
            quality_report.get("is_usable", False),
            live_embedding_stored_flag,
            primary_auth_method_log,
            primary_sim_log,
            primary_decision_log,
            benchmark_log_data,
            profile_updated_adaptively,
        )
        conn.commit()
        return jsonify(
            {
                "success": True,
                "authenticated": primary_authenticated,
                "user": {
                    "voiceId": voice_id,
                    "name": speaker_name_db,
                    "primary_method_similarity": float(primary_similarity),
                    "primary_method_used": stored_primary_method,
                },
                "message": auth_message,
                "benchmark_scores": benchmark_log_data,
                "live_audio_quality_passed": bool(
                    quality_report.get("is_usable", False)
                ),
                "profile_updated_adaptively": profile_updated_adaptively,
            }
        )
    except Exception as e:
        logging.error(
            f"Outer Auth error for {claimed_name if claimed_name else 'UnknownClaimedName'}:{e}",
            exc_info=True,
        )
        final_voice_id_log = (
            voice_id_log
            if "voice_id_log" in locals() and voice_id_log is not None
            else None
        )
        final_claimed_name_log = (
            claimed_name
            if "claimed_name" in locals() and claimed_name is not None
            else "UnknownDuringError"
        )
        final_quality_passed_log = (
            quality_report.get("is_usable", False)
            if "quality_report" in locals() and isinstance(quality_report, dict)
            else False
        )
        final_primary_method_log = (
            primary_auth_method_log
            if "primary_auth_method_log" in locals()
            and primary_auth_method_log is not None
            else None
        )
        final_primary_sim_log = (
            primary_sim_log
            if "primary_sim_log" in locals() and primary_sim_log is not None
            else None
        )
        final_primary_decision_log = (
            primary_decision_log
            if "primary_decision_log" in locals() and primary_decision_log is not None
            else None
        )
        final_benchmark_scores_log = (
            benchmark_log_data
            if benchmark_log_data
            else {"error_details": f"Server Exception:{str(e)[:100]}"}
        )
        final_profile_adapted_log = profile_updated_adaptively
        if conn and cursor:
            try:
                log_benchmark_attempt(
                    cursor,
                    final_voice_id_log,
                    final_claimed_name_log,
                    final_quality_passed_log,
                    False,
                    final_primary_method_log,
                    final_primary_sim_log,
                    final_primary_decision_log,
                    final_benchmark_scores_log,
                    final_profile_adapted_log,
                )
                conn.commit()
            except Exception as e_commit_log_on_error:
                logging.error(
                    f"Failed to commit error log for {final_claimed_name_log}:{e_commit_log_on_error}"
                )
        return (
            jsonify(
                {"success": False, "error": "Server authentication processing error."}
            ),
            500,
        )
    finally:
        if conn:
            conn.close()


@app.route(
    "/users", methods=["GET"]
)  # This route definition was missing from your last snippet
def get_users_route():
    conn = sqlite3.connect(DATABASE_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT voice_id, name, primary_aggregate_method, enrollment_timestamp, last_seen_timestamp FROM speaker_profiles ORDER BY enrollment_timestamp DESC"
    )
    users = [
        {
            "voiceId": r[0],
            "name": r[1],
            "agg_method": r[2],
            "enrolled": r[3],
            "last_seen": r[4],
        }
        for r in cur.fetchall()
    ]
    conn.close()
    return jsonify({"success": True, "users": users})


@app.route(
    "/benchmark-history/<string:username>", methods=["GET"]
)  # This route definition was missing
def benchmark_history_for_user(username):
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT voice_id FROM speaker_profiles WHERE name = ?", (username,)
        )
        user_profile = cursor.fetchone()
        if not user_profile:
            return jsonify({"success": False, "error": "User not found"}), 404
        user_voice_id = user_profile["voice_id"]
        cursor.execute(
            """SELECT log_id, authentication_timestamp, live_audio_quality_passed, primary_auth_method, primary_auth_similarity, primary_auth_decision, benchmark_scores_json, profile_was_adapted FROM authentication_benchmark_log WHERE speaker_voice_id = ? ORDER BY authentication_timestamp DESC""",
            (user_voice_id,),
        )
        logs = []
        for row in cursor.fetchall():
            log_entry = dict(row)
            try:
                log_entry["benchmark_scores"] = (
                    json.loads(log_entry["benchmark_scores_json"])
                    if log_entry["benchmark_scores_json"]
                    else {}
                )
            except json.JSONDecodeError:
                log_entry["benchmark_scores"] = {
                    "error": "Failed to parse benchmark JSON"
                }
            del log_entry["benchmark_scores_json"]
            logs.append(log_entry)
        return jsonify({"success": True, "username": username, "history": logs})
    except Exception as e:
        logging.error(f"Error benchmark history {username}: {e}", exc_info=True)
        return jsonify({"success": False, "error": "Server error history."}), 500
    finally:
        if conn:
            conn.close()


@app.route("/reset-db", methods=["POST"])  # This route definition was missing
def reset_db_route():
    try:
        if os.path.exists(DATABASE_PATH):
            os.remove(DATABASE_PATH)
        init_db()
        return jsonify({"success": True, "message": "Database reset successfully."})
    except Exception as e:
        logging.error(f"DB reset err:{e}", exc_info=True)
        return jsonify({"success": False, "error": "Failed to reset."}), 500


if __name__ == "__main__":
    port = config.FLASK_PORT  # Use from config
    debug_mode = config.FLASK_DEBUG_MODE  # Use from config
    logging.info(
        f"Starting Flask server on http://0.0.0.0:{port}, debug={debug_mode}..."
    )
    app.run(debug=debug_mode, host="0.0.0.0", port=port)

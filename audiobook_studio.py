import os
import json
import base64
import wave
import requests
import re
import time
import sys
import subprocess
import platform
import struct
import math
import concurrent.futures
import threading
import glob
import shutil

# --- DEPENDENCIES ---
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("\n[CRITICAL] 'numpy' library not found. Quality Control checks will be disabled.")
    print("Please run: pip install numpy")

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("\n[!] Warning: 'pydub' library not found. Output will be .WAV instead of .MP3.")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("\n[!] 'tqdm' not found. Progress bar disabled. (pip install tqdm)")

# --- CONSTANTS ---
MODELS = {
    "1": "gemini-2.5-flash-preview-tts", # High Limits
    "2": "gemini-2.5-pro-preview-tts"    # Strict Limit
}

# Recommended character limits based on model stability
RECOMMENDED_LIMITS = {
    "gemini-2.5-flash-preview-tts": 1500,
    "gemini-2.5-pro-preview-tts": 2400
}

# Daily Request Limits (RPD)
RPD_LIMITS = {
    "gemini-2.5-flash-preview-tts": 100, 
    "gemini-2.5-pro-preview-tts": 50     
}

VOICES = {
    "Male": {
        "1": ("Fenrir", "Deep/Imposing"),
        "2": ("Puck", "Young/Energetic"),
        "3": ("Orus", "Soft/Anxious"),
        "4": ("Charon", "Low/Steady"),
        "5": ("Enceladus", "Deep/Resonant")
    },
    "Female": {
        "1": ("Leda", "Standard/Warm"),
        "2": ("Aoede", "Soft/Elegant"),
        "3": ("Kore", "Clear/Bright"),
        "4": ("Callirrhoe", "Gentle/Calm"),
        "5": ("Zephyr", "Standard/Balanced")
    }
}

SAMPLE_RATE = 24000

# --- QUALITY CONTROL THRESHOLDS ---
QC_STRICT_SILENCE = 3.0
QC_ZCR_THRESHOLD = 0.20      # Hiss Detection
QC_RMS_STD_THRESHOLD = 150   # Monotone Detection
QC_PITCH_MALE_MAX = 175.0    # Above this? Likely female/child hallucination.
QC_PITCH_FEMALE_MIN = 135.0  # LOWERED to allow deeper female voices.

# --- GLOBAL THREADING CONTROL ---
# This is the critical fix for the "Machine Gun" effect.
# Regardless of how many workers are running, they must pass through this single gate.
API_LOCK = threading.Lock()
LAST_REQUEST_TIME = 0
MIN_REQUEST_INTERVAL = 2.0  # Minimum seconds between API hits

print_lock = threading.Lock()

def safe_print(msg):
    """Thread-safe print that plays nicely with TQDM progress bars."""
    with print_lock:
        if TQDM_AVAILABLE:
            tqdm.write(msg)
        else:
            print(msg)

def get_user_input(prompt, default=None):
    if default:
        user_in = input(f"{prompt} [{default}]: ").strip()
        return user_in if user_in else default
    else:
        while True:
            user_in = input(f"{prompt}: ").strip()
            if user_in:
                return user_in
            print("    -> This field is required.")

def play_audio_file(filepath):
    try:
        if platform.system() == 'Windows':
            os.startfile(filepath)
        elif platform.system() == 'Darwin':
            subprocess.call(('open', filepath))
        else:
            subprocess.call(('xdg-open', filepath))
    except Exception as e:
        safe_print(f"    [!] Could not auto-play. Please open '{filepath}' manually.")

def trim_silence(audio_bytes, threshold=80):
    try:
        total_len = len(audio_bytes)
        if total_len % 2 != 0: return audio_bytes
        scan_limit = min(total_len, 480000) 
        trim_index = total_len
        for i in range(total_len - 2, total_len - scan_limit, -2):
            sample = struct.unpack('<h', audio_bytes[i:i+2])[0]
            if abs(sample) > threshold:
                trim_index = min(total_len, i + 24000) 
                break
        if trim_index < total_len:
            return audio_bytes[:trim_index]
        return audio_bytes
    except:
        return audio_bytes

def estimate_fundamental_freq(audio_bytes, sample_rate=24000):
    if not NUMPY_AVAILABLE: return 0.0
    try:
        data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        N = len(data)
        if N < sample_rate * 0.5: return 0.0
        window = np.hanning(N)
        spectrum = np.abs(np.fft.rfft(data * window))
        freqs = np.fft.rfftfreq(N, 1/sample_rate)
        hps = np.copy(spectrum)
        for h in range(2, 4): 
            decimated = spectrum[::h]
            hps[:len(decimated)] *= decimated
        valid_idx = np.where((freqs > 60) & (freqs < 400))[0]
        if len(valid_idx) == 0: return 0.0
        peak_idx = valid_idx[np.argmax(hps[valid_idx])]
        return freqs[peak_idx]
    except Exception:
        return 0.0

def analyze_signal_metrics(audio_bytes, sample_rate=24000):
    if not NUMPY_AVAILABLE: return 0.0, 100.0, 1000.0
    try:
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
        if len(audio_data) == 0: return 0.0, 0.0, 0.0
        zero_crossings = np.sum(np.diff(np.signbit(audio_data).astype(int)) != 0)
        zcr = zero_crossings / len(audio_data)
        chunk_size = int(sample_rate * 0.1)
        n_chunks = len(audio_data) // chunk_size
        if n_chunks < 2: return zcr, 1000.0, 1000.0 
        truncated_len = n_chunks * chunk_size
        reshaped = audio_data[:truncated_len].reshape(n_chunks, chunk_size)
        rms_per_window = np.sqrt(np.mean(reshaped.astype(np.float64)**2, axis=1))
        avg_rms = np.mean(rms_per_window)
        rms_std_dev = np.std(rms_per_window)
        return zcr, avg_rms, rms_std_dev
    except Exception as e:
        safe_print(f"    [QC Error] Analysis failed: {e}")
        return 0.0, 1000.0, 1000.0

def check_audio_health(audio_bytes, text_len, target_gender="Male", threshold=100, max_silence_sec=2.0, extra_time=0.0):
    if not audio_bytes: return False, "Empty Data"
    total_samples = len(audio_bytes) // 2
    if total_samples == 0: return False, "Zero Samples"

    zcr, avg_rms, rms_std = analyze_signal_metrics(audio_bytes)
    pitch = estimate_fundamental_freq(audio_bytes)
    
    if zcr > QC_ZCR_THRESHOLD: return False, f"Metallic/Hissy Artifact (ZCR: {zcr:.2f})"
    if avg_rms < 50: return False, f"Low Volume (RMS: {int(avg_rms)})"
    if pitch > 0:
        if target_gender == "Male" and pitch > QC_PITCH_MALE_MAX:
            return False, f"Voice Drift Detected (Pitch High: {int(pitch)}Hz)"
        if target_gender == "Female" and pitch < QC_PITCH_FEMALE_MIN:
            return False, f"Voice Drift Detected (Pitch Low: {int(pitch)}Hz)"
    if rms_std < QC_RMS_STD_THRESHOLD: return False, f"Monotone/Flat Dynamics (StdDev: {int(rms_std)})"

    duration_sec = total_samples / SAMPLE_RATE
    MIN_CHARS_PER_SEC = 12.0
    max_allowed_duration = (text_len / MIN_CHARS_PER_SEC) + 10.0 + extra_time

    if duration_sec > max_allowed_duration:
        return False, f"Suspected Loop ({duration_sec:.1f}s > {max_allowed_duration:.1f}s limit)"
            
    return True, "OK"

def generate_audio_raw(text, voice_name, api_key, model_name):
    """
    Sends a request to Gemini API.
    CRITICAL FIX: Wraps the network call in a Global Lock to prevent Rate Limits.
    """
    if not text.strip(): return b""
    
    global LAST_REQUEST_TIME
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
    ]

    payload = {
        "contents": [{"parts": [{"text": text}]}],
        "safetySettings": safety_settings,
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice_name}}
            }
        }
    }
    
    # --- THROTTLING LOGIC ---
    # Even if 10 threads call this function, they must wait in single-file line here.
    with API_LOCK:
        elapsed = time.time() - LAST_REQUEST_TIME
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)
        
        LAST_REQUEST_TIME = time.time()
        
        # The request logic is inside the lock to ensure we don't start the timer 
        # until the previous request has actually fired.
        try:
            response = requests.post(url, json=payload, timeout=600)
            
            if response.status_code == 429: return "RATE_LIMIT"
            if response.status_code != 200: return f"API_ERR_{response.status_code}"
                
            data = response.json()
            if "candidates" not in data: return "NO_CANDIDATES"
            
            audio_b64 = data["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
            return base64.b64decode(audio_b64)

        except requests.exceptions.Timeout:
            return "EXCEPTION_TIMEOUT"
        except Exception as e:
            return f"EXCEPTION_{str(e)}"

def generate_audio_chunk(text, voice_name, api_key, model_name):
    """
    Directly sends text (with SSML tags) to the API.
    Native Mode: Trust the API to handle the tags.
    """
    res = generate_audio_raw(text, voice_name, api_key, model_name)
    if isinstance(res, str): return res # Propagate error
    if res: return trim_silence(res)
    return b""

# --- WORKER FUNCTION ---
def process_chunk_task(task_data):
    index, text, voice, key, model, output_dir, force_regen, gender_cat = task_data
    
    # Unified naming: .wav for friendly playback, %03d to catch standard indexes
    # Using 4 digits for future-proofing, but glob handles the match.
    filename = os.path.join(output_dir, f"chunk_{index:04d}.wav")
    text_filename = os.path.join(output_dir, f"chunk_{index:04d}.txt")
    
    # Smart Resume Check
    if not force_regen and os.path.exists(filename):
        is_valid_resume = False
        if os.path.exists(text_filename):
            try:
                with open(text_filename, "r", encoding="utf-8") as f:
                    cached_text = f.read()
                if cached_text == text:
                    is_valid_resume = True
            except: pass
        
        if is_valid_resume:
            if os.path.getsize(filename) > 0:
                return (index, True, filename, "Cached/Skipped")

    # Calculate intentional silence for QC only
    # (We no longer split by these tags, but we still need to know how long the audio *should* be)
    break_matches = re.findall(r'<break\s+time=[\'"]([\d\.]+)s[\'"]\s*/?>', text)
    total_break_time = sum(float(t) for t in break_matches)

    max_retries = 3
    text_len = len(text)
    
    for attempt in range(max_retries):
        result = generate_audio_chunk(text, voice, key, model)
        
        if isinstance(result, str):
            if "RATE_LIMIT" in result:
                safe_print(f"  [Worker {index+1}] Rate Limit Hit (429). Retrying...")
                time.sleep(10 + (attempt * 5))
                continue 
            else:
                safe_print(f"  [Worker {index+1}] API Error: {result}")
                time.sleep(2)
                continue

        if result:
            is_healthy, reason = check_audio_health(
                result, 
                text_len, 
                target_gender=gender_cat, 
                max_silence_sec=QC_STRICT_SILENCE,
                extra_time=total_break_time
            )
            
            if is_healthy:
                # Save as valid WAV for playback and resume compatibility
                with wave.open(filename, "wb") as w:
                    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SAMPLE_RATE)
                    w.writeframes(result)
                try:
                    with open(text_filename, "w", encoding="utf-8") as f: f.write(text)
                except: pass
                
                return (index, True, filename, "OK")
            else:
                safe_print(f"  [Worker {index+1}] QC Fail: {reason}. Retrying...")
                time.sleep(2)
                continue
        
    failed_filename = os.path.join(output_dir, f"chunk_{index:04d}_FAILED.wav")
    if result and isinstance(result, bytes):
        with wave.open(failed_filename, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SAMPLE_RATE)
            w.writeframes(result)
    
    return (index, False, failed_filename, "Max Retries Exceeded")

def smart_chunk_text(text, limit):
    """
    Tokenizes text by SSML tags to ensure we NEVER split inside a tag.
    Then accumulates tokens into chunks respecting the limit.
    """
    # Clean Markdown
    text = text.replace('**', '').replace('__', '')
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

    # 1. Split by tags (keeping tags).
    # This regex matches <...> sequences.
    tokens = re.split(r'(<[^>]+>)', text)
    
    chunks = []
    current_chunk = []
    current_len = 0
    
    for token in tokens:
        if not token: continue
        
        token_len = len(token)
        
        # Case A: Token is a Tag (Atomic, must not be split)
        if token.startswith('<') and token.endswith('>'):
            if current_len + token_len > limit and current_chunk:
                chunks.append("".join(current_chunk))
                current_chunk = []
                current_len = 0
            current_chunk.append(token)
            current_len += token_len
            
        # Case B: Token is Text (Can be split if needed)
        else:
            if current_len + token_len <= limit:
                current_chunk.append(token)
                current_len += token_len
            else:
                # Text is too big for the remaining space.
                # Split by natural boundaries: Newlines first, then sentence endings.
                # This regex splits by newline OR sentence ending (keeping the delimiter attached or separate).
                # (?<=[.?!]) matches position after punctuation.
                
                # Split into smaller pieces
                sub_parts = re.split(r'(\n|(?<=[.?!])\s+)', token)
                
                for part in sub_parts:
                    if not part: continue
                    part_len = len(part)
                    
                    if current_len + part_len > limit and current_chunk:
                        chunks.append("".join(current_chunk))
                        current_chunk = []
                        current_len = 0
                    
                    current_chunk.append(part)
                    current_len += part_len

    if current_chunk:
        chunks.append("".join(current_chunk))
        
    return chunks

def edit_text_in_external_editor(text):
    filename = "studio_quick_edit.txt"
    try:
        with open(filename, "w", encoding="utf-8") as f: f.write(text)
    except IOError: return text
    try:
        if platform.system() == 'Windows': os.startfile(filename)
        elif platform.system() == 'Darwin': subprocess.call(('open', filename))
        else: subprocess.call(('xdg-open', filename))
    except: pass
    input("\n[EDIT MODE] Edit the file, Save, then press ENTER here... ")
    try:
        with open(filename, "r", encoding="utf-8") as f: 
            new_text = f.read()
            try: os.remove(filename)
            except: pass
            return new_text.strip()
    except: return text

def select_option(options, label):
    print(f"\nSelect {label}:")
    for key, val in options.items():
        print(f"  {key}. {val[0]} ({val[1]})")
    while True:
        choice = input(f"Choose [1-{len(options)}]: ").strip()
        if choice in options: return options[choice][0]
        print("Invalid choice.")

def main():
    print("=============================")
    print("   GEMINI AUDIOBOOK STUDIO   ")
    print("=============================\n")

    if not NUMPY_AVAILABLE:
        print("[!] Numpy missing. Quality Control will be limited.")
    else:
        print(f"[QC Profile] ZCR Threshold: {QC_ZCR_THRESHOLD}")
        print(f"[QC Profile] Gender Sentry: Active")

    # --- PROJECT SETUP ---
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        api_key = get_user_input("Enter Gemini API Key")
    
    # Default to root 'chunks' folder to be friendly to previous script users
    chunks_dir = "chunks"
    os.makedirs(chunks_dir, exist_ok=True)
    
    # Resume Check
    # Look for both WAV (Parallel Script) and PCM (Studio Script)
    existing_chunks = glob.glob(os.path.join(chunks_dir, "chunk_*.wav")) + glob.glob(os.path.join(chunks_dir, "chunk_*.pcm"))
    resume_mode = False
    
    if existing_chunks:
        print(f"\n[!] Found {len(existing_chunks)} existing chunks.")
        choice = input("    [R]esume (Skip verified) | [O]verwrite (Delete all): ").lower().strip()
        if choice == 'o':
            # Delete both audio and text files
            for f in existing_chunks: os.remove(f)
            for f in glob.glob(os.path.join(chunks_dir, "chunk_*.txt")): os.remove(f)
        else:
            resume_mode = True

    input_file = get_user_input("Input Text File", "ln1_col1_hybrid.txt")
    raw_text = ""
    try:
        with open(input_file, "r", encoding="utf-8") as f: raw_text = f.read()
    except:
        print("File not found."); return

    # --- CONFIG ---
    print("\n--- Model ---")
    print("1. Gemini 2.5 Flash (Fast, High Limits)")
    print("2. Gemini 2.5 Pro (Better Context, Strict Limit)")
    m_choice = input("Choice [1]: ").strip()
    selected_model = MODELS.get(m_choice, MODELS["1"])
    
    # WORKER CONFIG
    max_workers = 4
    
    print("\n--- Narrator Selection ---")
    narrator_gender = input("Narrator Gender (m/f): ").lower()
    voice_cat = "Male" if narrator_gender.startswith('m') else "Female"
    reader_voice = select_option(VOICES[voice_cat], f"{voice_cat} Voice")
    
    # --- BUDGET OPTIMIZER LOOP ---
    current_limit = RECOMMENDED_LIMITS.get(selected_model, 1500)
    daily_limit = RPD_LIMITS.get(selected_model, 100)
    clean_text = raw_text.replace('“', '"').replace('”', '"')
    
    while True:
        print(f"\n[Configuration] Current Chunk Limit: {current_limit}")
        limit_str = input(f"Enter New Limit (Higher=Riskier) or Press [ENTER] to calculate: ").strip()
        
        if limit_str.isdigit():
            current_limit = int(limit_str)
        
        print("\n[Calculating Plan...]")
        chunks = smart_chunk_text(clean_text, current_limit)
        est_requests = len(chunks)
        safety_margin = daily_limit - est_requests
        
        print(f"--- STATISTICS ---")
        print(f"Total Chunks:   {est_requests}")
        print(f"Daily Limit:    {daily_limit}")
        print(f"Safety Margin:  {safety_margin} requests")
        
        choice = input("Accept plan and proceed? (y/n): ").lower().strip()
        if choice == 'y':
            break

    # Tasks definition (Pause Mode removed)
    tasks = [(i, chunks[i], reader_voice, api_key, selected_model, chunks_dir, not resume_mode, voice_cat) for i in range(len(chunks))]
    results = [None] * len(chunks) 
    
    # --- BATCH CONTROL ---
    print(f"\n[Batch Control] You have {len(tasks)} tasks queued.")
    batch_input = input("Press [ENTER] to run ALL, or enter a number to limit: ").strip()
    if batch_input.isdigit():
        limit_count = int(batch_input)
        tasks = tasks[:limit_count]

    from concurrent.futures import as_completed

    print(f"[Execution] Launching {max_workers} workers (Throttled Mode)...")
    print("------------------------------------------------")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for task in tasks:
            f = executor.submit(process_chunk_task, task)
            futures.append(f)
            time.sleep(0.1) 
            
        iterator = as_completed(futures)
        if TQDM_AVAILABLE:
            iterator = tqdm(iterator, total=len(tasks), unit="chunk", desc="Processing", ncols=100)
            
        for future in iterator:
            idx, success, fname, msg = future.result()
            if idx < len(results):
                results[idx] = (success, fname, msg)
            
            status = "OK" if success else "FAIL"
            
            if not TQDM_AVAILABLE:
                if msg != "Cached/Skipped":
                    safe_print(f"[Chunk {idx+1}] {status}: {msg}")

    # --- REVIEW PHASE ---
    director_mode = input("\nEnable Director Review (Check/Retry files)? (y/n) [y]: ").lower() != 'n'
    segment_files = []
    
    for i in range(len(tasks)):
        if not results[i]: continue 
        
        while True:
            success, fname, msg = results[i]
            
            if not success:
                print(f"\n[!] Chunk {i+1} FAILED: {msg}")
                has_file = fname and os.path.exists(fname)
                print("Options: [R]etry | [E]dit Text | [D]iscard", end="")
                if has_file: print(" | [L]isten", end="")
                print("")
                
                choice = input("Select: ").lower().strip()
                if choice == 'd': break 
                elif choice == 'l' and has_file:
                    play_audio_file(fname)
                elif choice in ('r', 'e'):
                    if choice == 'e':
                        chunks[i] = edit_text_in_external_editor(chunks[i])
                    print("    Regenerating...")
                    t_res = process_chunk_task((i, chunks[i], reader_voice, api_key, selected_model, chunks_dir, True, voice_cat))
                    results[i] = (t_res[1], t_res[2], t_res[3])
            
            else:
                if director_mode:
                    if "Cached" in msg:
                        segment_files.append(fname)
                        break

                    print(f"\nReviewing Chunk {i+1} ({msg})...")
                    play_audio_file(fname)
                    
                    choice = input("    [K]eep | [R]etry | [E]dit: ").lower().strip()
                    if not choice or choice == 'k':
                        segment_files.append(fname); break 
                    elif choice in ('r', 'e'):
                        if choice == 'e':
                            chunks[i] = edit_text_in_external_editor(chunks[i])
                        t_res = process_chunk_task((i, chunks[i], reader_voice, api_key, selected_model, chunks_dir, True, voice_cat))
                        results[i] = (t_res[1], t_res[2], t_res[3])
                else:
                    segment_files.append(fname)
                    break

    # --- STITCHING PHASE ---
    if segment_files:
        print(f"\nStitching {len(segment_files)} segments...")
        temp = "temp_master.wav"
        with wave.open(temp, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SAMPLE_RATE)
            for seg in segment_files:
                try:
                    with wave.open(seg, "rb") as f: w.writeframes(f.readframes(f.getnframes()))
                except: pass
        
        output_filename = "final_audiobook"
        final = f"{output_filename}.mp3" if PYDUB_AVAILABLE else f"{output_filename}.wav"
        
        if PYDUB_AVAILABLE:
            AudioSegment.from_wav(temp).export(final, format="mp3", bitrate="192k")
            os.remove(temp)
        else:
            os.rename(temp, final)
        print(f"\n[SUCCESS] Saved to {final}")

if __name__ == "__main__":
    main()
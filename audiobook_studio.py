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

# Pitch Thresholds (Hz) - Used to detect gender swaps
QC_PITCH_MALE_MAX = 175.0    # Above this? Likely female/child hallucination.
QC_PITCH_FEMALE_MIN = 155.0  # Below this? Likely male hallucination.

# Thread-safe printing
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

def generate_silence(duration_sec):
    num_samples = int(SAMPLE_RATE * duration_sec)
    return b'\x00\x00' * num_samples

def estimate_fundamental_freq(audio_bytes, sample_rate=24000):
    """
    Estimates the fundamental frequency (pitch) using Harmonic Product Spectrum.
    Useful for detecting if a male voice suddenly becomes female (pitch jump).
    """
    if not NUMPY_AVAILABLE: return 0.0
    
    try:
        # Convert PCM to float array
        data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        N = len(data)
        if N < sample_rate * 0.5: return 0.0 # Too short to analyze accurately
        
        # Windowing to reduce spectral leakage
        window = np.hanning(N)
        
        # FFT
        spectrum = np.abs(np.fft.rfft(data * window))
        freqs = np.fft.rfftfreq(N, 1/sample_rate)
        
        # HPS (Harmonic Product Spectrum) - Downsample and multiply to find fundamental
        hps = np.copy(spectrum)
        for h in range(2, 4): 
            decimated = spectrum[::h]
            hps[:len(decimated)] *= decimated
            
        # Limit search to human voice range (50Hz - 400Hz)
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

        # ZCR
        zero_crossings = np.sum(np.diff(np.signbit(audio_data).astype(int)) != 0)
        zcr = zero_crossings / len(audio_data)

        # Dynamic Range
        chunk_size = int(sample_rate * 0.1)
        n_chunks = len(audio_data) // chunk_size
        
        if n_chunks < 2:
            return zcr, 1000.0, 1000.0 
        
        truncated_len = n_chunks * chunk_size
        reshaped = audio_data[:truncated_len].reshape(n_chunks, chunk_size)
        rms_per_window = np.sqrt(np.mean(reshaped.astype(np.float64)**2, axis=1))
        
        avg_rms = np.mean(rms_per_window)
        rms_std_dev = np.std(rms_per_window)
        
        return zcr, avg_rms, rms_std_dev
        
    except Exception as e:
        safe_print(f"    [QC Error] Analysis failed: {e}")
        return 0.0, 1000.0, 1000.0

def check_audio_health(audio_bytes, text_len, target_gender="Male", threshold=100, max_silence_sec=2.0):
    if not audio_bytes: return False, "Empty Data"
    total_samples = len(audio_bytes) // 2
    if total_samples == 0: return False, "Zero Samples"

    zcr, avg_rms, rms_std = analyze_signal_metrics(audio_bytes)
    pitch = estimate_fundamental_freq(audio_bytes)
    
    # 1. Artifact Checks
    if zcr > QC_ZCR_THRESHOLD: 
        return False, f"Metallic/Hissy Artifact (ZCR: {zcr:.2f})"
    if avg_rms < 50: 
        return False, f"Low Volume (RMS: {int(avg_rms)})"
    
    # 2. Gender/Pitch Consistency Check
    if pitch > 0:
        if target_gender == "Male" and pitch > QC_PITCH_MALE_MAX:
            return False, f"Voice Drift Detected (Pitch High: {int(pitch)}Hz > {int(QC_PITCH_MALE_MAX)}Hz)"
        if target_gender == "Female" and pitch < QC_PITCH_FEMALE_MIN:
            return False, f"Voice Drift Detected (Pitch Low: {int(pitch)}Hz < {int(QC_PITCH_FEMALE_MIN)}Hz)"

    # 3. Monotone Check
    if rms_std < QC_RMS_STD_THRESHOLD:
        return False, f"Monotone/Flat Dynamics (StdDev: {int(rms_std)})"

    # 4. Loop Check
    duration_sec = total_samples / SAMPLE_RATE
    MIN_CHARS_PER_SEC = 12.0
    # RELAXED THRESHOLD: Increased buffer from 5.0 to 10.0 to prevent false flags on slow reads
    max_allowed_duration = (text_len / MIN_CHARS_PER_SEC) + 10.0

    if duration_sec > max_allowed_duration:
        return False, f"Suspected Loop ({duration_sec:.1f}s > {max_allowed_duration:.1f}s limit)"
            
    return True, "OK"

def generate_audio_raw(text, voice_name, api_key, model_name):
    if not text.strip(): return b""
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
    
    try:
        # TIMEOUT INCREASED: 600 seconds (10 mins) to prevent timeouts on Pro model
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

def generate_audio_with_pauses(text, voice_name, api_key, model_name):
    pattern = r'<break\s+time=[\'"]([\d\.]+)s[\'"]\s*/?>'
    parts = re.split(pattern, text)
    combined_audio = bytearray()
    
    i = 0
    while i < len(parts):
        segment_text = parts[i]
        
        if segment_text.strip():
            res = generate_audio_raw(segment_text, voice_name, api_key, model_name)
            if isinstance(res, str): return res
            if res: combined_audio.extend(trim_silence(res))
        
        if i + 1 < len(parts):
            try:
                duration = float(parts[i+1])
                combined_audio.extend(generate_silence(duration))
            except ValueError: pass
            i += 1
        i += 1
        
    return bytes(combined_audio)

# --- WORKER FUNCTION ---
def process_chunk_task(task_data):
    index, text, voice, key, model, output_dir, force_regen, gender_cat = task_data
    
    filename = os.path.join(output_dir, f"chunk_{index:04d}.pcm")
    
    # RESUME CHECK
    if not force_regen and os.path.exists(filename):
        if os.path.getsize(filename) > 0:
            return (index, True, filename, "Cached/Skipped")

    max_retries = 3
    text_len = len(text)
    
    for attempt in range(max_retries):
        result = generate_audio_with_pauses(text, voice, key, model)
        
        if isinstance(result, str):
            if "RATE_LIMIT" in result:
                safe_print(f"  [Worker {index+1}] Rate Limit Hit. Sleeping...")
                time.sleep(10 + (attempt * 5))
                continue 
            else:
                safe_print(f"  [Worker {index+1}] API Error: {result}")
                time.sleep(2)
                continue

        if result:
            # PASS GENDER TO QC
            is_healthy, reason = check_audio_health(result, text_len, target_gender=gender_cat, max_silence_sec=QC_STRICT_SILENCE)
            
            if is_healthy:
                with open(filename, "wb") as f: f.write(result)
                return (index, True, filename, "OK")
            else:
                safe_print(f"  [Worker {index+1}] QC Fail: {reason}. Retrying ({attempt+1}/{max_retries})...")
                time.sleep(2)
                continue
        
    failed_filename = os.path.join(output_dir, f"chunk_{index:04d}_FAILED.pcm")
    if result and isinstance(result, bytes):
        with open(failed_filename, "wb") as f: f.write(result)
    
    return (index, False, failed_filename, "Max Retries Exceeded")

def smart_chunk_text(text, limit):
    chunks = []
    current_chunk = ""
    lines = text.split('\n')
    
    header_pattern = re.compile(r'^#{1,6}\s+')
    
    for line in lines:
        line = line.replace('**', '').replace('__', '') 
        line = header_pattern.sub('', line)
        
        if len(current_chunk) + len(line) < limit:
            current_chunk += "\n" + line
        else:
            if current_chunk: chunks.append(current_chunk)
            current_chunk = line
    if current_chunk: chunks.append(current_chunk)
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
        print(f"[QC Profile] ZCR Threshold: {QC_ZCR_THRESHOLD} (Hiss Detection)")
        print(f"[QC Profile] RMS Dynamic Range: {QC_RMS_STD_THRESHOLD} (Monotone Detection)")
        print(f"[QC Profile] Gender Sentry Active (Male > {int(QC_PITCH_MALE_MAX)}Hz / Female < {int(QC_PITCH_FEMALE_MIN)}Hz)")

    # --- PROJECT SETUP ---
    api_key = get_user_input("Enter Gemini API Key")
    
    project_name = get_user_input("Project Name (Folder)", "MyAudiobook")
    base_dir = os.path.join("output", project_name)
    chunks_dir = os.path.join(base_dir, "chunks")
    
    os.makedirs(chunks_dir, exist_ok=True)
    
    # Resume Check
    existing_chunks = glob.glob(os.path.join(chunks_dir, "chunk_*.pcm"))
    resume_mode = False
    
    if existing_chunks:
        print(f"\n[!] Found {len(existing_chunks)} existing chunks in '{chunks_dir}'.")
        choice = input("    [R]esume (Skip generated) | [O]verwrite (Delete all): ").lower().strip()
        if choice == 'o':
            print("    Deleting old chunks...")
            for f in existing_chunks: os.remove(f)
        else:
            resume_mode = True
            print("    Resume mode enabled. Existing valid chunks will be skipped.")

    input_file = get_user_input("Input Text File", "el_standard.txt")
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
    
    # Set worker config based on model type
    if "flash" in selected_model:
        max_workers = 4
        stagger_delay = 0.5 
    else:
        max_workers = 4 
        stagger_delay = 15.0  # REVERTED to 15.0 for Stability (Rate Limits count as requests)

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
        print(f"Daily Limit:    {daily_limit} (Model: {selected_model})")
        print(f"Safety Margin:  {safety_margin} requests (Spare room for retries)")
        
        if safety_margin < 0:
            print(f"[!] WARNING: This plan exceeds your daily limit by {abs(safety_margin)} requests.")
        elif safety_margin < 5:
            print(f"[!] CAUTION: Very tight margin ({safety_margin}). Retries may hit limit.")
        else:
            print(f"[OK] Plan looks safe.")
            
        choice = input("Accept plan and proceed? (y/n): ").lower().strip()
        if choice == 'y':
            break
        print("    -> Adjusting limit...")

    # ADD GENDER TO TASK TUPLE
    tasks = [(i, chunks[i], reader_voice, api_key, selected_model, chunks_dir, not resume_mode, voice_cat) for i in range(len(chunks))]
    results = [None] * len(chunks) 
    
    # --- BATCH CONTROL ---
    print(f"\n[Batch Control] You have {len(tasks)} tasks queued.")
    batch_input = input("Press [ENTER] to run ALL, or enter a number (e.g. 50) to limit this run: ").strip()
    if batch_input.isdigit():
        limit_count = int(batch_input)
        tasks = tasks[:limit_count]
        print(f"    -> Run limited to first {limit_count} tasks. (Run again tomorrow to resume the rest)")

    # Need to import as_completed for the progress bar logic
    from concurrent.futures import as_completed

    print(f"[Execution] Launching {max_workers} workers...")
    print("------------------------------------------------")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        print("Initializing tasks...")
        for task in tasks:
            f = executor.submit(process_chunk_task, task)
            futures.append(f)
            time.sleep(0.1 if resume_mode else stagger_delay)
            
        iterator = as_completed(futures)
        
        if TQDM_AVAILABLE:
            iterator = tqdm(iterator, total=len(tasks), unit="chunk", desc="Processing", ncols=100)
            
        for future in iterator:
            idx, success, fname, msg = future.result()
            if idx < len(results):
                results[idx] = (success, fname, msg)
            
            status = "OK" if success else "FAIL"
            
            if TQDM_AVAILABLE:
                if not success or (msg != "Cached/Skipped"):
                    tqdm.write(f"[Chunk {idx+1}] {status}: {msg}")
            else:
                if msg != "Cached/Skipped":
                    safe_print(f"[Completed] Chunk {idx+1}: {status} ({msg})")

    # --- REVIEW PHASE ---
    director_mode = input("\nEnable Director Review (Check/Retry files)? (y/n) [y]: ").lower() != 'n'
    review_all = False
    
    if director_mode and resume_mode:
        review_all = input("    Review cached/skipped files too? (y/n) [n]: ").lower() == 'y'
    
    print("\n--- Final Assembly Review ---")
    segment_files = []
    
    processed_count = len(tasks)
    
    for i in range(processed_count):
        if not results[i]: continue 
        
        while True:
            success, fname, msg = results[i]
            
            # CASE 1: FAILED
            if not success:
                print(f"\n[!] Chunk {i+1} FAILED: {msg}")
                has_file = fname and os.path.exists(fname)
                
                print("Options: [R]etry Sync | [E]dit Text | [D]iscard", end="")
                if has_file: print(" | [L]isten Failed", end="")
                print("")
                
                choice = input("Select: ").lower().strip()
                if not choice: choice = 'r'

                if choice == 'd':
                    print("    Discarding.")
                    break 
                
                elif choice == 'l' and has_file:
                    with open(fname, "rb") as f: raw = f.read()
                    with wave.open("preview.wav", "wb") as w:
                        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SAMPLE_RATE); w.writeframes(raw)
                    play_audio_file("preview.wav")
                    if input("    Force Keep? (y/n): ").lower() == 'y':
                        results[i] = (True, fname, "Manual Override") 
                
                elif choice == 'e':
                    new_text = edit_text_in_external_editor(chunks[i])
                    chunks[i] = new_text
                    print("    Regenerating...")
                    # Pass voice_cat here as well
                    t_res = process_chunk_task((i, chunks[i], reader_voice, api_key, selected_model, chunks_dir, True, voice_cat))
                    results[i] = (t_res[1], t_res[2], t_res[3])

                elif choice == 'r':
                    print("    Regenerating...")
                    # Pass voice_cat here as well
                    t_res = process_chunk_task((i, chunks[i], reader_voice, api_key, selected_model, chunks_dir, True, voice_cat))
                    results[i] = (t_res[1], t_res[2], t_res[3])
            
            # CASE 2: SUCCESS
            else:
                if director_mode:
                    if "Cached" in msg and not review_all:
                        segment_files.append(fname)
                        break

                    print(f"\nReviewing Chunk {i+1} ({msg})...")
                    
                    with open(fname, "rb") as f: current_raw = f.read()
                    preview_data = current_raw
                    
                    if segment_files:
                        try:
                            prev_file = segment_files[-1]
                            with open(prev_file, "rb") as f: prev_raw = f.read()
                            
                            bytes_to_take = 3 * 24000 * 2
                            context = prev_raw[-bytes_to_take:] if len(prev_raw) > bytes_to_take else prev_raw
                            spacer = b'\x00\x00' * 12000 # 0.5s silence
                            preview_data = context + spacer + current_raw
                            print("    (Playing with 3s context from previous clip...)")
                        except: pass

                    with wave.open("preview.wav", "wb") as w:
                        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SAMPLE_RATE); w.writeframes(preview_data)
                    play_audio_file("preview.wav")
                    
                    choice = input("    [K]eep | [R]etry | [E]dit Text | [D]iscard: ").lower().strip()
                    if not choice: choice = 'k'
                    
                    if choice == 'k':
                        segment_files.append(fname)
                        break 
                    elif choice == 'd':
                        print("    Discarding.")
                        break 
                    elif choice == 'e':
                        new_text = edit_text_in_external_editor(chunks[i])
                        chunks[i] = new_text
                        print("    Regenerating...")
                        t_res = process_chunk_task((i, chunks[i], reader_voice, api_key, selected_model, chunks_dir, True, voice_cat))
                        results[i] = (t_res[1], t_res[2], t_res[3])
                    elif choice == 'r':
                        print("    Regenerating...")
                        t_res = process_chunk_task((i, chunks[i], reader_voice, api_key, selected_model, chunks_dir, True, voice_cat))
                        results[i] = (t_res[1], t_res[2], t_res[3])
                else:
                    segment_files.append(fname)
                    break

    # --- STITCHING PHASE ---
    if segment_files:
        print(f"\nStitching {len(segment_files)} segments...")
        temp = os.path.join(base_dir, "temp_master.wav")
        with wave.open(temp, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SAMPLE_RATE)
            for seg in segment_files:
                try:
                    with open(seg, "rb") as f: w.writeframes(f.read())
                except: pass
        
        output_filename = os.path.join(base_dir, "final_audiobook")
        final = f"{output_filename}.mp3" if PYDUB_AVAILABLE else f"{output_filename}.wav"
        
        if PYDUB_AVAILABLE:
            AudioSegment.from_wav(temp).export(final, format="mp3", bitrate="192k")
            os.remove(temp)
        else:
            os.rename(temp, final)
        print(f"\n[SUCCESS] Saved to {final}")

if __name__ == "__main__":
    main()
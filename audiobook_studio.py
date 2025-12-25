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

CHUNK_LIMIT = 3000
SAMPLE_RATE = 24000

# --- QUALITY CONTROL THRESHOLDS ---
QC_STRICT_SILENCE = 3.0
QC_ZCR_THRESHOLD = 0.20      # Default: 0.20. (0.25=Robots only, 0.18-0.20=Hiss/Static)
QC_RMS_STD_THRESHOLD = 150   # Default: 150. (Higher=Stricter on Monotone)

# Thread-safe printing
print_lock = threading.Lock()

def safe_print(msg):
    with print_lock:
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

def check_audio_health(audio_bytes, text_len, threshold=100, max_silence_sec=2.0):
    if not audio_bytes: return False, "Empty Data"
    total_samples = len(audio_bytes) // 2
    if total_samples == 0: return False, "Zero Samples"

    zcr, avg_rms, rms_std = analyze_signal_metrics(audio_bytes)
    
    if zcr > QC_ZCR_THRESHOLD: 
        return False, f"Metallic/Hissy Artifact (ZCR: {zcr:.2f})"
    if avg_rms < 50: 
        return False, f"Low Volume (RMS: {int(avg_rms)})"
    if rms_std < QC_RMS_STD_THRESHOLD:
        return False, f"Monotone/Flat Dynamics (StdDev: {int(rms_std)})"

    duration_sec = total_samples / SAMPLE_RATE
    MIN_CHARS_PER_SEC = 12.0
    max_allowed_duration = (text_len / MIN_CHARS_PER_SEC) + 5.0

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
        # TIMEOUT ADDED: 120 seconds to account for long generation queues
        response = requests.post(url, json=payload, timeout=120)
        
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
    index, text, voice, key, model, output_dir, force_regen = task_data
    
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
            is_healthy, reason = check_audio_health(result, text_len, max_silence_sec=QC_STRICT_SILENCE)
            
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
    
    # Compile regex for Markdown headers (e.g. "# Chapter 1" or "## The End")
    header_pattern = re.compile(r'^#{1,6}\s+')
    
    for line in lines:
        # Pre-clean formatting
        line = line.replace('**', '').replace('__', '') 
        # Remove markdown header symbols so TTS doesn't say "Hashtag"
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
    
    if "flash" in selected_model:
        max_workers = 4
        stagger_delay = 0.5 
    else:
        max_workers = 4 
        stagger_delay = 15.0 

    print("\n--- Narrator Selection ---")
    narrator_gender = input("Narrator Gender (m/f): ").lower()
    voice_cat = "Male" if narrator_gender.startswith('m') else "Female"
    reader_voice = select_option(VOICES[voice_cat], f"{voice_cat} Voice")
    
    # --- PROCESSING ---
    print("\n[Normalizing Text...]")
    clean_text = raw_text.replace('“', '"').replace('”', '"')
    chunks = smart_chunk_text(clean_text, CHUNK_LIMIT)
    
    tasks = [(i, chunks[i], reader_voice, api_key, selected_model, chunks_dir, not resume_mode) for i in range(len(chunks))]
    results = [None] * len(chunks) 
    
    # Need to import as_completed for the progress bar logic
    from concurrent.futures import as_completed

    print(f"\n[Plan] Book split into {len(chunks)} chunks.")
    print(f"[Execution] Launching {max_workers} workers...")
    print("------------------------------------------------")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        # 1. SUBMISSION PHASE
        # We submit sequentially to respect the stagger delay (preventing instant rate limits)
        print("Initializing tasks...")
        for task in tasks:
            f = executor.submit(process_chunk_task, task)
            futures.append(f)
            # Stagger submissions to avoid hitting API rate limits instantly
            time.sleep(0.1 if resume_mode else stagger_delay)
            
        # 2. MONITORING PHASE
        # We wrap as_completed so the bar updates as threads finish, regardless of order
        iterator = as_completed(futures)
        
        if TQDM_AVAILABLE:
            iterator = tqdm(iterator, total=len(chunks), unit="chunk", desc="Processing", ncols=100)
            
        for future in iterator:
            idx, success, fname, msg = future.result()
            results[idx] = (success, fname, msg) # Store result in correct index
            
            status = "OK" if success else "FAIL"
            
            # If using TQDM, use .write() to print above the bar without breaking it
            # If valid/cached, we stay silent to keep UI clean, unless it failed.
            if TQDM_AVAILABLE:
                if not success or (msg != "Cached/Skipped"):
                    tqdm.write(f"[Chunk {idx+1}] {status}: {msg}")
            else:
                # Fallback for no TQDM
                if msg != "Cached/Skipped":
                    safe_print(f"[Completed] Chunk {idx+1}: {status} ({msg})")

    # --- REVIEW PHASE ---
    director_mode = input("\nEnable Director Review (Check/Retry files)? (y/n) [y]: ").lower() != 'n'
    review_all = False
    
    if director_mode and resume_mode:
        review_all = input("    Review cached/skipped files too? (y/n) [n]: ").lower() == 'y'
    
    print("\n--- Final Assembly Review ---")
    segment_files = []
    
    for i in range(len(chunks)):
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
                    t_res = process_chunk_task((i, chunks[i], reader_voice, api_key, selected_model, chunks_dir, True))
                    results[i] = (t_res[1], t_res[2], t_res[3])

                elif choice == 'r':
                    print("    Regenerating...")
                    t_res = process_chunk_task((i, chunks[i], reader_voice, api_key, selected_model, chunks_dir, True))
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
                    
                    # Contextual Preview
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
                        t_res = process_chunk_task((i, chunks[i], reader_voice, api_key, selected_model, chunks_dir, True))
                        results[i] = (t_res[1], t_res[2], t_res[3])
                    elif choice == 'r':
                        print("    Regenerating...")
                        t_res = process_chunk_task((i, chunks[i], reader_voice, api_key, selected_model, chunks_dir, True))
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
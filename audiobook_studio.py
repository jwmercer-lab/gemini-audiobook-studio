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

# Try to import pydub
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("\n[!] Warning: 'pydub' library not found. Output will be .WAV instead of .MP3.")

# --- CONSTANTS ---
MODELS = {
    "1": "gemini-2.5-flash-preview-tts", # High Limits
    "2": "gemini-2.5-pro-preview-tts"    # Strict Limit
}

# Single Voice Selection
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
QC_STRICT_SILENCE = 3.0

# Thread-safe printing
print_lock = threading.Lock()

def safe_print(msg):
    with print_lock:
        print(msg)

def cleanup_temp_files():
    """Removes old PCM chunks to prevent contamination."""
    files = glob.glob("chunk_*.pcm")
    count = 0
    for f in files:
        try:
            os.remove(f)
            count += 1
        except: pass
    if count > 0:
        print(f"[Startup] Cleaned up {count} old chunk files.")

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

def check_audio_health(audio_bytes, text_len, threshold=100, max_silence_sec=2.0):
    """
    Analyzes audio for:
    1. Low Volume (RMS)
    2. Dead Air (Silence)
    3. Hallucination Loops (Duration vs Text Length)
    """
    if not audio_bytes: return False, "Empty Data"
    total_samples = len(audio_bytes) // 2
    if total_samples == 0: return False, "Zero Samples"

    # --- 1. DURATION CHECK (Loop Detection) ---
    duration_sec = total_samples / SAMPLE_RATE
    
    # Heuristic: English narration is rarely slower than 12 chars/sec.
    # If duration is significantly longer than text warrants, it's likely repeating.
    # We add a small buffer (5s) to avoid flagging very short valid clips.
    MIN_CHARS_PER_SEC = 12.0
    max_allowed_duration = (text_len / MIN_CHARS_PER_SEC) + 5.0

    if duration_sec > max_allowed_duration:
        return False, f"Suspected Loop ({duration_sec:.1f}s > {max_allowed_duration:.1f}s limit)"

    # --- 2. SILENCE & VOLUME CHECK ---
    current_silence_run = 0
    longest_silence_run = 0
    sum_squares = 0
    
    try:
        fmt = f"<{total_samples}h"
        samples = struct.unpack(fmt, audio_bytes)
        
        for sample in samples:
            val = abs(sample)
            sum_squares += val * val
            
            if val < threshold:
                current_silence_run += 1
            else:
                if current_silence_run > longest_silence_run:
                    longest_silence_run = current_silence_run
                current_silence_run = 0
        
        # Catch tail silence
        if current_silence_run > longest_silence_run:
            longest_silence_run = current_silence_run
            
        rms = math.sqrt(sum_squares / total_samples)
        if rms < 50: 
            return False, f"Low Volume (RMS: {int(rms)})"
            
        silence_duration = longest_silence_run / SAMPLE_RATE
        if silence_duration > max_silence_sec:
            return False, f"Dead Air Detected ({silence_duration:.2f}s)"
            
        return True, "OK"
    except Exception as e:
        return False, f"Analysis Error: {e}"

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
        response = requests.post(url, json=payload)
        if response.status_code == 429: return "RATE_LIMIT"
        if response.status_code != 200: return f"API_ERR_{response.status_code}"
            
        data = response.json()
        if "candidates" not in data: return "NO_CANDIDATES"
        
        audio_b64 = data["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
        return base64.b64decode(audio_b64)
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
    index, text, voice, key, model = task_data
    max_retries = 3
    
    # Calculate text length for the loop check
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
            # PASS TEXT LEN HERE vvv
            is_healthy, reason = check_audio_health(result, text_len, max_silence_sec=QC_STRICT_SILENCE)
            
            if is_healthy:
                fname = f"chunk_{index:04d}.pcm"
                with open(fname, "wb") as f: f.write(result)
                return (index, True, fname, "OK")
            else:
                safe_print(f"  [Worker {index+1}] QC Fail: {reason}. Retrying ({attempt+1}/{max_retries})...")
                if attempt == max_retries - 1:
                    fname = f"chunk_{index:04d}_FAILED.pcm"
                    with open(fname, "wb") as f: f.write(result)
                    return (index, False, fname, f"QC Failed: {reason}")
                time.sleep(2)
                continue
        
    return (index, False, None, "Max Retries Exceeded")

def smart_chunk_text(text, limit):
    chunks = []
    current_chunk = ""
    lines = text.split('\n')
    for line in lines:
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

    cleanup_temp_files()

    api_key = get_user_input("Enter Gemini API Key")
    input_file = get_user_input("Input Text File", "el_standard.txt")
    raw_text = ""
    try:
        with open(input_file, "r", encoding="utf-8") as f: raw_text = f.read()
    except:
        print("File not found."); return

    output_name = get_user_input("Output Filename", "audiobook_final")
    
    # --- MODEL ---
    print("\n--- Model ---")
    print("1. Gemini 2.5 Flash (Fast, High Limits)")
    print("2. Gemini 2.5 Pro (Better Context, Strict Limit)")
    m_choice = input("Choice [1]: ").strip()
    selected_model = MODELS.get(m_choice, MODELS["1"])
    
    # --- STAGGER SETTINGS ---
    if "flash" in selected_model:
        max_workers = 4
        stagger_delay = 0.5 
    else:
        max_workers = 4 
        stagger_delay = 15.0 

    # --- VOICE SETUP ---
    print("\n--- Narrator Selection ---")
    narrator_gender = input("Narrator Gender (m/f): ").lower()
    voice_cat = "Male" if narrator_gender.startswith('m') else "Female"
    reader_voice = select_option(VOICES[voice_cat], f"{voice_cat} Voice")
    
    print("\n[Normalizing Text...]")
    clean_text = raw_text.replace('“', '"').replace('”', '"')
    chunks = smart_chunk_text(clean_text, CHUNK_LIMIT)
    
    print(f"\n[Plan] Book split into {len(chunks)} chunks.")
    print(f"[Execution] Launching {max_workers} workers with {stagger_delay}s stagger...")
    print("------------------------------------------------")

    tasks = [(i, chunks[i], reader_voice, api_key, selected_model) for i in range(len(chunks))]
    results = [None] * len(chunks) 
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for task in tasks:
            f = executor.submit(process_chunk_task, task)
            futures.append(f)
            safe_print(f"  -> Submitted Worker {task[0]+1}. Waiting {stagger_delay}s...")
            time.sleep(stagger_delay)
            
        for i, future in enumerate(futures):
            idx, success, fname, msg = future.result()
            results[idx] = (success, fname, msg)
            status = "OK" if success else "FAIL"
            safe_print(f"[Completed] Chunk {idx+1}: {status} ({msg})")

    # --- REVIEW PHASE ---
    director_mode = input("\nEnable Director Review (Check/Retry files)? (y/n) [y]: ").lower() != 'n'
    
    print("\n--- Final Assembly Review ---")
    segment_files = []
    
    for i in range(len(chunks)):
        # Infinite Loop for Current Chunk (allows multiple retries)
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
                    break # Break loop, move to next chunk, do NOT add to segment_files
                
                elif choice == 'l' and has_file:
                    with open(fname, "rb") as f: raw = f.read()
                    with wave.open("preview.wav", "wb") as w:
                        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SAMPLE_RATE); w.writeframes(raw)
                    play_audio_file("preview.wav")
                    if input("    Force Keep? (y/n): ").lower() == 'y':
                        results[i] = (True, fname, "Manual Override") # Update state to Success
                    # Loop continues, now state is Success
                
                elif choice == 'e':
                    new_text = edit_text_in_external_editor(chunks[i])
                    chunks[i] = new_text
                    print("    Regenerating...")
                    res = generate_audio_with_pauses(chunks[i], reader_voice, api_key, selected_model)
                    if res and isinstance(res, bytes):
                        fname = f"chunk_{i:04d}.pcm"
                        with open(fname, "wb") as f: f.write(res)
                        # UPDATE: Pass len
                        is_healthy, reason = check_audio_health(res, len(chunks[i]), max_silence_sec=QC_STRICT_SILENCE)
                        results[i] = (is_healthy, fname, "OK" if is_healthy else reason)
                    else:
                        results[i] = (False, None, f"API Error: {res}")

                elif choice == 'r':
                    print("    Regenerating...")
                    res = generate_audio_with_pauses(chunks[i], reader_voice, api_key, selected_model)
                    if res and isinstance(res, bytes):
                        fname = f"chunk_{i:04d}.pcm"
                        with open(fname, "wb") as f: f.write(res)
                        # UPDATE: Pass len
                        is_healthy, reason = check_audio_health(res, len(chunks[i]), max_silence_sec=QC_STRICT_SILENCE)
                        results[i] = (is_healthy, fname, "OK" if is_healthy else reason)
                    else:
                        results[i] = (False, None, f"API Error: {res}")
            
            # CASE 2: SUCCESS
            else:
                if director_mode:
                    print(f"\nReviewing Chunk {i+1} ({msg})...")
                    with open(fname, "rb") as f: raw = f.read()
                    with wave.open("preview.wav", "wb") as w:
                        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SAMPLE_RATE); w.writeframes(raw)
                    play_audio_file("preview.wav")
                    
                    choice = input("    [K]eep | [R]etry | [E]dit Text | [D]iscard: ").lower().strip()
                    if not choice: choice = 'k'
                    
                    if choice == 'k':
                        segment_files.append(fname)
                        break # Done with this chunk
                    elif choice == 'd':
                        print("    Discarding.")
                        break # Done
                    elif choice == 'e':
                        new_text = edit_text_in_external_editor(chunks[i])
                        chunks[i] = new_text
                        print("    Regenerating...")
                        res = generate_audio_with_pauses(chunks[i], reader_voice, api_key, selected_model)
                        if res and isinstance(res, bytes):
                            with open(fname, "wb") as f: f.write(res)
                            # UPDATE: Pass len
                            is_healthy, reason = check_audio_health(res, len(chunks[i]), max_silence_sec=QC_STRICT_SILENCE)
                            results[i] = (is_healthy, fname, "OK" if is_healthy else reason)
                    elif choice == 'r':
                        print("    Regenerating...")
                        res = generate_audio_with_pauses(chunks[i], reader_voice, api_key, selected_model)
                        if res and isinstance(res, bytes):
                            with open(fname, "wb") as f: f.write(res)
                            # UPDATE: Pass len
                            is_healthy, reason = check_audio_health(res, len(chunks[i]), max_silence_sec=QC_STRICT_SILENCE)
                            results[i] = (is_healthy, fname, "OK" if is_healthy else reason)
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
                    with open(seg, "rb") as f: w.writeframes(f.read())
                    os.remove(seg)
                except: pass
        
        final = f"{output_name}.mp3" if PYDUB_AVAILABLE else f"{output_name}.wav"
        if PYDUB_AVAILABLE:
            AudioSegment.from_wav(temp).export(final, format="mp3", bitrate="192k")
            os.remove(temp)
        else:
            os.rename(temp, final)
        print(f"\n[SUCCESS] Saved to {final}")

if __name__ == "__main__":
    main()
import sys
import wave
import getpass
import requests
import base64
import os
import shutil
import time
import subprocess
import platform

# Securely grab the API key
secure_api_key = getpass.getpass("Enter your API Key: ")

# --- NEW: Single vs Multi-Speaker Selection ---
mode = input("Single voice or Multi-speaker? (s/m): ").strip().lower()

director_mode = input("Enable Director Mode to review chunks as they generate? (y/n): ").strip().lower() == 'y'

model_input = input("Select model (2.5-pro / 3.1-flash) [default: 2.5-pro]: ").strip().lower()
if '3.1' in model_input or 'flash' in model_input:
    model_name = "gemini-3.1-flash-tts-preview"
else:
    model_name = "gemini-2.5-pro-tts"

if mode == 'm':
    spk1_name = input("Enter Speaker 1 Name (as written in text, e.g., Sarah): ").strip()
    spk1_voice = input("Enter Speaker 1 Voice (e.g., Leda): ").strip()
    spk2_name = input("Enter Speaker 2 Name (as written in text, e.g., Sam): ").strip()
    spk2_voice = input("Enter Speaker 2 Voice (e.g., Puck): ").strip()
    filename = input("Enter the name of the text file: ")
    voice_choice = f"Multi_{spk1_voice}_{spk2_voice}" 
else:
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = input("Enter the name of the text file: ")
    voice_choice = input("Enter the Voice Name (e.g., Fenrir, Charon, Puck): ")

# Vertex AI REST endpoint
url = f"https://us-central1-aiplatform.googleapis.com/v1/projects/gen-lang-client-0158533571/locations/us-central1/publishers/google/models/{model_name}:generateContent?key={secure_api_key}"

# Variable style prompt
style_prompt = input("Enter style prompt (or press Enter to skip): ").strip()

try:
    with open(filename, "r", encoding="utf-8") as file:
        poem_text = file.read()
except FileNotFoundError:
    print(f"Error: '{filename}' not found.")
    sys.exit(1)

# --- Smart Chunking Logic ---
raw_chunks = poem_text.split("***")
chunks = [c.strip() for c in raw_chunks if c.strip()]

print(f"File loaded. Found {len(chunks)} chunk(s).")

# --- Pre-Flight Check ---
prepared_chunks = []
oversized = []

for index, chunk in enumerate(chunks, 1):
    if style_prompt:
        combined_contents = f"{style_prompt}: {chunk}"
    else:
        combined_contents = chunk
        
    char_count = len(combined_contents)
    prepared_chunks.append((index, combined_contents, char_count))
    
    if char_count > 3000:
        oversized.append((index, char_count, chunk))

if oversized:
    print("\n[!] Pre-Flight Warning: Oversized chunks detected.")
    for idx, count, _ in oversized:
        print(f"    - Chunk {idx}: {count} characters")
        
    proceed = input("\nDo you want to proceed anyway? (y/n): ").strip().lower()
    if proceed != 'y':
        print("\n--- OVERSIZED CHUNK DUMP ---")
        for idx, count, text in oversized:
            print(f"\n[ Chunk {idx} | {count} characters ]\n")
            print(text)
        print("\n-----------------------------")
        print("Aborted. Edit the text and try again.")
        sys.exit(1)

print("\nPre-flight complete.")

# --- Caching Setup ---
base_name = os.path.splitext(filename)[0]
cache_dir = f".tts_cache_{base_name}_{voice_choice}"
os.makedirs(cache_dir, exist_ok=True)

print(f"Cache directory engaged: {cache_dir}\n")

def play_audio(filepath):
    try:
        if platform.system() == 'Windows':
            os.startfile(filepath)
        elif platform.system() == 'Darwin':
            subprocess.call(('open', filepath))
        else:
            subprocess.call(('xdg-open', filepath))
    except:
        print(f"    [!] Could not auto-play '{filepath}'. Open it manually to review.")

# --- Generation & Caching Loop ---
for index, combined_contents, char_count in prepared_chunks:
    chunk_wav_path = os.path.join(cache_dir, f"chunk_{index:03d}.wav")
    chunk_txt_path = os.path.join(cache_dir, f"chunk_{index:03d}.txt")

    # Check for existing cached chunk
    if os.path.exists(chunk_wav_path) and os.path.exists(chunk_txt_path):
        with open(chunk_txt_path, "r", encoding="utf-8") as f:
            cached_text = f.read()
        
        if cached_text == combined_contents:
            print(f"Skipping chunk {index}/{len(prepared_chunks)} - Valid cache found.")
            continue

    chunk_approved = False
    while not chunk_approved:
        print(f"Transmitting chunk {index}/{len(prepared_chunks)} ({char_count} chars)...")

        # Dynamic Speech Config
        if mode == 'm':
        payload = {
            "contents": [{"role": "user", "parts": [{"text": combined_contents}]}],
            "generationConfig": {
                "temperature": 1.0,
                "speechConfig": speech_config
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}
            ]
        }

        # Rate Limit Survival Loop
        max_retries = 5
        for attempt in range(max_retries):
            response = requests.post(url, json=payload)
            
            if response.status_code == 429:
                wait_time = 10 + (attempt * 5)
                print(f"    [!] Rate limit hit (429). Cooling down for {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            elif response.status_code != 200:
                print(f"API Error on chunk {index} - Code {response.status_code}: {response.text}")
                print("Run aborted. Previous chunks are safely cached.")
                sys.exit(1)
            break
        else:
            print("Max rate limit retries exceeded. Aborting.")
            sys.exit(1)

        response_data = response.json()

        try:
            base64_audio = response_data['candidates'][0]['content']['parts'][0]['inlineData']['data']
            raw_audio_data = base64.b64decode(base64_audio)

            # Write the audio chunk to cache
            with wave.open(chunk_wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(raw_audio_data)

            # Write the text verification card to cache
            with open(chunk_txt_path, "w", encoding="utf-8") as f:
                f.write(combined_contents)

        except (KeyError, IndexError):
            print(f"Safety filter tripped or invalid response format on chunk {index}.")
            print("Raw Response:", response_data)
            sys.exit(1)

        if not director_mode:
            chunk_approved = True
        else:
            play_audio(chunk_wav_path)
            print(f"\n[ Director Mode | Chunk {index} ]")
            choice = input("    [K]eep or [R]etry? (k/r): ").strip().lower()
            if choice == 'r':
                print("    Discarding and retrying...")
                time.sleep(1) # Breathe before hitting the API again
            else:
                chunk_approved = True

# --- Final Assembly ---
output_filename = f"{base_name}_{voice_choice}.wav"
print(f"\nAll chunks complete. Stitching master file...")

with wave.open(output_filename, "wb") as master_wf:
    master_wf.setnchannels(1)
    master_wf.setsampwidth(2)
    master_wf.setframerate(24000)
    
    for index, _, _ in prepared_chunks:
        chunk_wav_path = os.path.join(cache_dir, f"chunk_{index:03d}.wav")
        with wave.open(chunk_wav_path, "rb") as chunk_wf:
            master_wf.writeframes(chunk_wf.readframes(chunk_wf.getnframes()))

print(f"Execution complete. Audio saved as: {output_filename}")

# --- Cleanup ---
cleanup = input(f"\nDo you want to delete the temporary cache folder ({cache_dir})? (y/n): ").strip().lower()
if cleanup == 'y':
    shutil.rmtree(cache_dir)
    print("Cache wiped.")
else:
    print("Cache retained.")
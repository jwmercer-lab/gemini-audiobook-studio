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
import tempfile

# Securely grab the API key
secure_api_key = getpass.getpass("Enter your API Key: ")

# --- Single vs Multi-Speaker Selection ---
mode = input("Single voice or Multi-speaker? (s/m): ").strip().lower()

director_mode = input("Enable Director Mode to review chunks as they generate? (y/n): ").strip().lower() == 'y'

model_input = input("Select model (3.1-flash / 2.5-pro) [default: 3.1-flash]: ").strip().lower()
if '2.5' in model_input or 'pro' in model_input:
    model_name = "gemini-2.5-pro-tts"
    char_limit = 3000
else:
    model_name = "gemini-3.1-flash-tts-preview"
    char_limit = 2000

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
        poem_text = file.read().replace('\ufeff', '')
except FileNotFoundError:
    print(f"Error: '{filename}' not found.")
    sys.exit(1)

# --- Smart Chunking Logic (Initial Baseline) ---
raw_chunks = poem_text.split("***")
initial_chunks = [c.strip() for c in raw_chunks if c.strip()]
total_chunks = len(initial_chunks)

print(f"File loaded. Baseline structure contains {total_chunks} chunk(s).")

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

def edit_text_in_editor(initial_text):
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode='w', encoding='utf-8') as tf:
        tf.write(initial_text)
        temp_path = tf.name
    
    try:
        sys_platform = platform.system()
        if sys_platform == 'Windows':
            subprocess.run(["notepad", temp_path])
        elif sys_platform == 'Darwin':
            subprocess.run(["open", "-W", "-e", temp_path])
        else:
            gui_editors = ['xed', 'gedit', 'mousepad', 'kate']
            launched = False
            for editor in gui_editors:
                if shutil.which(editor):
                    subprocess.Popen([editor, temp_path], stderr=subprocess.DEVNULL)
                    launched = True
                    break
            
            if not launched:
                fallback = os.environ.get('EDITOR', 'nano')
                subprocess.run([fallback, temp_path])
                
        print("\n    [->] Editor window deployed.")
        input("    [->] Make your revisions, SAVE the file, then press ENTER here to sync text... ")
                
        with open(temp_path, 'r', encoding='utf-8') as tf:
            updated_text = tf.read()
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return updated_text

def normalize_text(text):
    if text is None:
        return ""
    return text.replace("\r\n", "\n").strip()

# --- Generation & Caching Loop ---
index = 1
while index <= total_chunks:
    chunk_wav_path = os.path.join(cache_dir, f"chunk_{index:03d}.wav")
    chunk_txt_path = os.path.join(cache_dir, f"chunk_{index:03d}.txt")
    chunk_orig_path = os.path.join(cache_dir, f"chunk_{index:03d}.orig.txt")

    # --- Live Disk Read & Validation Mechanism ---
    while True:
        try:
            with open(filename, "r", encoding="utf-8") as file:
                live_text = file.read().replace('\ufeff', '')
        except FileNotFoundError:
            print(f"Error: Master file '{filename}' vanished mid-run.")
            sys.exit(1)
        
        live_chunks = [c.strip() for c in live_text.split("***") if c.strip()]
        
        if index > len(live_chunks):
            print(f"\n[!] Structural Break: Expected chunk {index}, but file only contains {len(live_chunks)} chunks.")
            print("    You may have accidentally deleted a '***' delimiter during an external edit.")
            input("    Restore the file structure on disk and press ENTER to re-read... ")
            continue
            
        raw_chunk = live_chunks[index - 1]
        combined_contents = f"{style_prompt}: {raw_chunk}" if style_prompt else raw_chunk
        char_count = len(combined_contents)
        
        # Immediate size audit on the live text
        if char_count > char_limit:
            print(f"\n[!] Size Alert: Live text for Chunk {index} contains {char_count} characters (Limit: {char_limit}).")
            print("    [O]verride and force transmission anyway")
            print("    [E]dit this chunk right now using the windowed editor")
            print("    [R]eload master file (Fix the text in your external editor first)")
            live_action = input("Select action (o/e/r): ").strip().lower()
            
            if live_action == 'e':
                edited_block = edit_text_in_editor(combined_contents)
                
                # Inline split handler for size alert edits
                if "***" in edited_block:
                    sub_chunks = [c.strip() for c in edited_block.split("***") if c.strip()]
                    if len(sub_chunks) > 1:
                        print(f"\n[+] Split Detected: Dividing chunk {index} into {len(sub_chunks)} components.")
                        processed_subs = []
                        for sc in sub_chunks:
                            if style_prompt and sc.startswith(f"{style_prompt}: "):
                                sc = sc[len(style_prompt) + 2:]
                            processed_subs.append(sc)
                        
                        live_chunks[index - 1 : index] = processed_subs
                        with open(filename, "w", encoding="utf-8") as file:
                            file.write("\n\n***\n\n".join(live_chunks))
                        total_chunks = len(live_chunks)
                        
                        edited_block = sub_chunks[0]
                        if style_prompt and not edited_block.startswith(f"{style_prompt}: "):
                            edited_block = f"{style_prompt}: {edited_block}"
                
                combined_contents = edited_block
                char_count = len(combined_contents)
                break
            elif live_action == 'o':
                break
            else:
                continue
        else:
            break

    source_ingest_text = combined_contents

    # Flexible Dual-Match Cache Validation Logic
    if os.path.exists(chunk_wav_path) and os.path.exists(chunk_txt_path):
        with open(chunk_txt_path, "r", encoding="utf-8") as f:
            cached_edited = normalize_text(f.read())
            
        cached_orig = ""
        if os.path.exists(chunk_orig_path):
            with open(chunk_orig_path, "r", encoding="utf-8") as f:
                cached_orig = normalize_text(f.read())

        target_compare = normalize_text(source_ingest_text)

        if target_compare == cached_edited or (cached_orig and target_compare == cached_orig):
            with open(chunk_txt_path, "r", encoding="utf-8") as f:
                combined_contents = f.read()
            char_count = len(combined_contents)
            print(f"Skipping chunk {index}/{total_chunks} - Valid match found (master file aligns with cache).")
            index += 1
            continue

    chunk_approved = False
    needs_generation = True

    while not chunk_approved:
        if needs_generation:
            print(f"Transmitting chunk {index}/{total_chunks} ({char_count} chars)...")

            if mode == 'm':
                speech_config = {
                    "languageCode": "en-US",
                    "multiSpeakerVoiceConfig": {
                        "speakerVoiceConfigs": [
                            {"speaker": spk1_name, "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": spk1_voice}}},
                            {"speaker": spk2_name, "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": spk2_voice}}}
                        ]
                    }
                }
            else:
                speech_config = {
                    "languageCode": "en-US",
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {"voiceName": voice_choice}
                    }
                }

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

                with wave.open(chunk_wav_path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(24000)
                    wf.writeframes(raw_audio_data)

                with open(chunk_txt_path, "w", encoding="utf-8") as f:
                    f.write(combined_contents)
                
                with open(chunk_orig_path, "w", encoding="utf-8") as f:
                    f.write(source_ingest_text)
                
                needs_generation = False

            except (KeyError, IndexError):
                print(f"\n[!] Safety filter tripped or invalid response format on chunk {index}.")
                print("Raw Response:", response_data)
                
                filter_choice = input("\n    Do you want to [E]dit this chunk to bypass the filter or [A]bort? (e/a): ").strip().lower()
                if filter_choice == 'e':
                    editing = True
                    current_edit_text = combined_contents
                    re_submit = False
                    while editing:
                        updated_text = edit_text_in_editor(current_edit_text)
                        
                        # Inline split handler for filter edits
                        if "***" in updated_text:
                            sub_chunks = [c.strip() for c in updated_text.split("***") if c.strip()]
                            if len(sub_chunks) > 1:
                                print(f"\n[+] Split Detected: Dividing chunk {index} into {len(sub_chunks)} components.")
                                with open(filename, "r", encoding="utf-8") as file:
                                    file_text = file.read().replace('\ufeff', '')
                                live_file_chunks = [c.strip() for c in file_text.split("***") if c.strip()]
                                
                                processed_subs = []
                                for sc in sub_chunks:
                                    if style_prompt and sc.startswith(f"{style_prompt}: "):
                                        sc = sc[len(style_prompt) + 2:]
                                    processed_subs.append(sc)
                                
                                live_file_chunks[index - 1 : index] = processed_subs
                                with open(filename, "w", encoding="utf-8") as file:
                                    file.write("\n\n***\n\n".join(live_file_chunks))
                                total_chunks = len(live_file_chunks)
                                
                                updated_text = sub_chunks[0]
                                if style_prompt and not updated_text.startswith(f"{style_prompt}: "):
                                    updated_text = f"{style_prompt}: {updated_text}"

                        new_char_count = len(updated_text)
                        within_limit = new_char_count <= char_limit
                        
                        print(f"\n[ Filter Edit Review | Chunk {index} ]")
                        print(f"    Character count: {new_char_count} / {char_limit}")
                        if within_limit:
                            print("    Status: Within safe operational limits.")
                        else:
                            print("    Status: [!] WARNING: Exceeds model character limit.")
                        
                        print("\n    Options:")
                        print("        [S]ubmit adjusted text to cloud")
                        print("        [E]dit the chunk again")
                        print("        [A]bort run entirely")
                        
                        post_edit_choice = input("\nSelect an option (s/e/a): ").strip().lower()
                        
                        if post_edit_choice == 's':
                            if not within_limit:
                                confirm = input("    [!] Text exceeds limit. Force submission anyway? (y/n): ").strip().lower()
                                if confirm != 'y':
                                    current_edit_text = updated_text
                                    continue
                            
                            combined_contents = updated_text
                            char_count = new_char_count
                            re_submit = True
                            editing = False
                        elif post_edit_choice == 'a':
                            print("Aborted by user during filter resolution.")
                            sys.exit(1)
                        else:
                            current_edit_text = updated_text
                    
                    if re_submit:
                        continue
                else:
                    print("Run aborted. Previous chunks are safely cached.")
                    sys.exit(1)

        if not director_mode:
            chunk_approved = True
        else:
            play_audio(chunk_wav_path)
            print(f"\n[ Director Mode | Chunk {index} ]")
            choice = input("    [K]eep, [R]etry, or [E]dit? (k/r/e): ").strip().lower()
            
            if choice == 'r':
                print("    Discarding and retrying...")
                needs_generation = True
                time.sleep(1)
            elif choice == 'e':
                editing = True
                current_edit_text = combined_contents
                while editing:
                    updated_text = edit_text_in_editor(current_edit_text)
                    
                    # Inline split handler for standard Director Mode edits
                    if "***" in updated_text:
                        sub_chunks = [c.strip() for c in updated_text.split("***") if c.strip()]
                        if len(sub_chunks) > 1:
                            print(f"\n[+] Split Detected: Dividing chunk {index} into {len(sub_chunks)} components.")
                            with open(filename, "r", encoding="utf-8") as file:
                                file_text = file.read().replace('\ufeff', '')
                            live_file_chunks = [c.strip() for c in file_text.split("***") if c.strip()]
                            
                            processed_subs = []
                            for sc in sub_chunks:
                                if style_prompt and sc.startswith(f"{style_prompt}: "):
                                    sc = sc[len(style_prompt) + 2:]
                                processed_subs.append(sc)
                            
                            live_file_chunks[index - 1 : index] = processed_subs
                            with open(filename, "w", encoding="utf-8") as file:
                                file.write("\n\n***\n\n".join(live_file_chunks))
                            total_chunks = len(live_file_chunks)
                            
                            updated_text = sub_chunks[0]
                            if style_prompt and not updated_text.startswith(f"{style_prompt}: "):
                                updated_text = f"{style_prompt}: {updated_text}"

                    new_char_count = len(updated_text)
                    within_limit = new_char_count <= char_limit
                    
                    print(f"\n[ Edit Review | Chunk {index} ]")
                    print(f"    Character count: {new_char_count} / {char_limit}")
                    if within_limit:
                        print("    Status: Within safe operational limits.")
                    else:
                        print("    Status: [!] WARNING: Exceeds model character limit.")
                    
                    print("\n    Options:")
                    print("        [S]ubmit to cloud and generate audio")
                    print("        [E]dit the chunk again")
                    print("        [A]bort edit and revert to previous text")
                    
                    post_edit_choice = input("\nSelect an option (s/e/a): ").strip().lower()
                    
                    if post_edit_choice == 's':
                        if not within_limit:
                            confirm = input("    [!] Text exceeds limit. Force submission anyway? (y/n): ").strip().lower()
                            if confirm != 'y':
                                current_edit_text = updated_text
                                continue
                        
                        combined_contents = updated_text
                        char_count = new_char_count
                        needs_generation = True
                        editing = False
                    elif post_edit_choice == 'a':
                        print("    Edit preserved in volatile memory but discarded from pipeline.")
                        editing = False
                    else:
                        current_edit_text = updated_text
            else:
                chunk_approved = True

    # Advance pointer only after structural validation and verification are complete
    index += 1

# --- Final Assembly ---
output_filename = f"{base_name}_{voice_choice}.wav"
output_text_filename = f"{base_name}_revised.txt"
print(f"\nAll chunks complete. Stitching master files...")

# Stitch Audio directly from expanded disk footprint footprint
with wave.open(output_filename, "wb") as master_wf:
    master_wf.setnchannels(1)
    master_wf.setsampwidth(2)
    master_wf.setframerate(24000)
    
    for i in range(1, total_chunks + 1):
        chunk_wav_path = os.path.join(cache_dir, f"chunk_{i:03d}.wav")
        with wave.open(chunk_wav_path, "rb") as chunk_wf:
            master_wf.writeframes(chunk_wf.readframes(chunk_wf.getnframes()))

# Stitch Text directly from expanded disk footprint footprint
final_text_chunks = []
for i in range(1, total_chunks + 1):
    chunk_txt_path = os.path.join(cache_dir, f"chunk_{i:03d}.txt")
    with open(chunk_txt_path, "r", encoding="utf-8") as f:
        final_text_chunks.append(f.read())

with open(output_text_filename, "w", encoding="utf-8") as f:
    f.write("\n\n***\n\n".join(final_text_chunks))

print(f"Execution complete.")
print(f"-> Audio saved as: {output_filename}")
print(f"-> Revised text saved as: {output_text_filename}")

cleanup = input(f"\nKeep the temporary cache folder ({cache_dir}) for future edits? [Y/n]: ").strip().lower()
if cleanup == 'n':
    shutil.rmtree(cache_dir)
    print("Cache wiped.")
else:
    print("Cache retained.")
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
import re
import argparse

API_BASE_URL = os.environ.get(
    "GEMINI_API_ENDPOINT",
    "https://generativelanguage.googleapis.com/v1beta"
)

DEFAULT_MODEL_38 = "gemini-3.8-flash-tts"
DEFAULT_MODEL_25 = "gemini-2.5-pro-preview-tts"

# Operational character thresholds
CHAR_LIMIT_38_CHUNKED = 4500
CHAR_LIMIT_38_SINGLE_TAKE = 8000
CHAR_LIMIT_25 = 3000
DEFAULT_PAUSE_SEC = 1.5

def resolve_api_key(cli_key=None):
    """Retrieves the Gemini API Key from CLI flags, env vars, or secure prompt."""
    if cli_key:
        return cli_key.strip()
    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_key:
        print("[+] API Key loaded securely from GEMINI_API_KEY environment variable.")
        return env_key
    print("\nNo API key detected in environment (GEMINI_API_KEY).")
    prompted = getpass.getpass("Enter your Google Gemini API Key: ").strip()
    if not prompted:
        print("Error: A valid Gemini API Key is required to continue.")
        sys.exit(1)
    return prompted

def play_audio(filepath):
    """Spawns the native platform media player to audition an audio take."""
    try:
        if platform.system() == 'Windows':
            os.startfile(filepath)
        elif platform.system() == 'Darwin':
            subprocess.call(('open', filepath))
        else:
            subprocess.call(('xdg-open', filepath))
    except Exception as exc:
        print(f"    [!] Could not auto-play '{filepath}': {exc}. Review manually.")

def edit_text_in_editor(initial_text):
    """Launches the user's OS native text editor for in-flight script revisions."""
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

        print("\n    [->] Editor window spawned.")
        input("    [->] Make revisions, SAVE the file, then press ENTER in this terminal to sync... ")

        with open(temp_path, 'r', encoding='utf-8') as tf:
            updated_text = tf.read()
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return updated_text

def normalize_text(text):
    """Normalizes line endings for deterministic dual-match caching."""
    if text is None:
        return ""
    return text.replace("\r\n", "\n").strip()

def clean_text_for_tts(text):
    """
    Sanitizes script text before payload transmission to prevent read-aloud errors:
    - Strips code/developer comments starting with '#'
    - Strips bracketed stage directions [...] so non-verbal foley cues are not spoken
    - Converts unsupported <long pause> tags into valid <short pause> tags
    - Strips trailing pause tags at the end of chunks
    """
    if not text:
        return ""
    lines = [line for line in text.splitlines() if not line.strip().startswith('#')]
    cleaned = "\n".join(lines)
    # Strip bracketed cues across single and multiple lines
    cleaned = re.sub(r'\[.*?\]', '', cleaned, flags=re.DOTALL)
    # Replace invalid <long pause> tags that cause literal read-aloud bugs in 3.8
    cleaned = re.sub(r'<\s*long pause\s*>', '<short pause>', cleaned, flags=re.IGNORECASE)
    # Strip redundant trailing pause tags at the end of sections
    cleaned = re.sub(r'<\s*(?:short\s+)?pause\s*>\s*$', '', cleaned, flags=re.IGNORECASE)
    return cleaned.strip()

def build_request_parts(chunk_text, mode, spk1_name=None, spk2_name=None, style=None, insert_lead_pause=False, default_speaker=None):
    """
    Parses dialogue text into Gemini 3.8 multi-turn content parts.
    Strips speaker name prefixes (e.g. 'Narrator:', 'Sam:') from verbatim dialogue
    and routes them out-of-band via speech_metadata.speaker and style.
    """
    cleaned_chunk = clean_text_for_tts(chunk_text)
    if not cleaned_chunk:
        return []

    parts = []

    if mode == 'm' and spk1_name and spk2_name:
        s1_lower = spk1_name.strip().lower()
        s2_lower = spk2_name.strip().lower()

        # Matches lines beginning with "Speaker:" or "Speaker -"
        speaker_line_pattern = re.compile(
            rf"^(?:<\s*short pause\s*>\s*)?({re.escape(spk1_name)}|{re.escape(spk2_name)})\s*[:\-]\s*(.*)$",
            re.IGNORECASE
        )
        speaker_standalone_pattern = re.compile(
            rf"^(?:<\s*short pause\s*>\s*)?({re.escape(spk1_name)}|{re.escape(spk2_name)})\s*$",
            re.IGNORECASE
        )

        lines = cleaned_chunk.split('\n')
        current_speaker = default_speaker or spk1_name
        current_buffer = []

        for line in lines:
            trimmed = line.strip()
            if not trimmed:
                continue

            # Case 1: Standalone speaker tag on its own line
            m_standalone = speaker_standalone_pattern.match(trimmed)
            if m_standalone:
                matched_speaker = m_standalone.group(1).strip()
                new_speaker = spk1_name if matched_speaker.lower() == s1_lower else spk2_name
                if current_buffer:
                    combined_block = " ".join(current_buffer).strip()
                    if combined_block:
                        meta = {"speaker": current_speaker}
                        if style:
                            meta["style"] = style
                        parts.append({"text": combined_block, "speech_metadata": meta})
                    current_buffer = []
                current_speaker = new_speaker
                continue

            # Case 2: Speaker tag prefixed directly on spoken dialogue
            m_line = speaker_line_pattern.match(trimmed)
            if m_line:
                matched_speaker = m_line.group(1).strip()
                new_speaker = spk1_name if matched_speaker.lower() == s1_lower else spk2_name
                dialogue = m_line.group(2).strip()

                if current_buffer:
                    combined_block = " ".join(current_buffer).strip()
                    if combined_block:
                        meta = {"speaker": current_speaker}
                        if style:
                            meta["style"] = style
                        parts.append({"text": combined_block, "speech_metadata": meta})
                    current_buffer = []

                current_speaker = new_speaker
                if dialogue:
                    current_buffer.append(dialogue)
                continue

            # Case 3: Regular narrative line
            current_buffer.append(trimmed)

        if current_buffer:
            combined_block = " ".join(current_buffer).strip()
            if combined_block:
                meta = {"speaker": current_speaker}
                if style:
                    meta["style"] = style
                parts.append({"text": combined_block, "speech_metadata": meta})

        if not parts:
            meta = {"speaker": current_speaker}
            if style:
                meta["style"] = style
            parts.append({"text": cleaned_chunk, "speech_metadata": meta})
    else:
        part_obj = {"text": cleaned_chunk}
        if style:
            part_obj["speech_metadata"] = {"style": style}
        parts.append(part_obj)

    # Insert breath pause at start of turn if requested
    if insert_lead_pause and parts:
        first_text = parts[0]["text"]
        if not first_text.startswith("<"):
            parts[0]["text"] = f"<short pause> {first_text}"

    return parts

def send_tts_request(url, payload, max_retries=5):
    """Posts generation requests to the Gemini API with exponential backoff on 429."""
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=120)
        except requests.exceptions.RequestException as e:
            print(f"    [!] Connection error ({e}). Retrying in 5s...")
            time.sleep(5)
            continue

        if response.status_code == 429:
            wait_time = 12 + (attempt * 6)
            print(f"    [!] Rate limit reached (429). Backing off for {wait_time}s...")
            time.sleep(wait_time)
            continue
        elif response.status_code == 403:
            print(f"\n[!] API Error [HTTP 403]: {response.text}")
            print("    Diagnosis: Permission Denied. Ensure your key was created at https://aistudio.google.com/apikey")
            return None
        elif response.status_code != 200:
            print(f"\n[!] API Error [HTTP {response.status_code}]: {response.text}")
            return None
        return response
    print("    [!] Max rate-limit retries exceeded.")
    return None

def save_audio_payload(data_bytes, destination_wav_path):
    """
    Saves decoded audio bytes to disk, detecting RIFF WAV containers
    and wrapping raw PCM frames as needed.
    """
    if data_bytes.startswith(b"RIFF"):
        with open(destination_wav_path, "wb") as f:
            f.write(data_bytes)
    else:
        with wave.open(destination_wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(data_bytes)

def parse_arguments():
    """Configures CLI flags with fallback to friendly interactive setup."""
    parser = argparse.ArgumentParser(
        description="Gemini 3.8 Flash TTS Audiobook Production Studio"
    )
    parser.add_argument("script", nargs="?", help="Path to manuscript text file (.txt)")
    parser.add_argument("--key", help="Gemini API Key (optional, defaults to GEMINI_API_KEY)")
    parser.add_argument("--workflow", choices=["chunked", "single-take"], default=None,
                        help="Production workflow: chunked (Director review) or single-take (full master)")
    parser.add_argument("--mode", choices=["s", "m"], default=None,
                        help="Voice mode: s (single voice) or m (multi-speaker)")
    parser.add_argument("--model", choices=["3.8-flash", "2.5-pro"], default=None,
                        help="TTS model engine (default: 3.8-flash)")
    parser.add_argument("--style", default="", help="Global performance style prompt")
    parser.add_argument("--pause", type=float, default=DEFAULT_PAUSE_SEC,
                        help=f"Silence between scenes in seconds for chunked mode (default: {DEFAULT_PAUSE_SEC})")
    return parser.parse_args()

def main():
    args = parse_arguments()
    secure_api_key = resolve_api_key(args.key)

    print("\n=============================================")
    print("   GEMINI AUDIOBOOK PRODUCTION STUDIO")
    print("   Engine: Gemini 3.8 Flash TTS / 2.5 Pro")
    print("=============================================\n")

    # Workflow selection
    if args.workflow == "single-take":
        single_take_mode = True
    elif args.workflow == "chunked":
        single_take_mode = False
    else:
        print("Select Production Workflow:")
        print("  [1] Chunked Director Mode (Audition & approve each vignette individually)")
        print("  [2] Full Story Single-Take (Generate entire manuscript in one seamless pass)")
        wf_input = input("Select workflow (1/2) [default: 1]: ").strip() or '1'
        single_take_mode = (wf_input == '2')

    # Speaker mode selection
    if args.mode:
        mode = args.mode
    else:
        mode = input("Single voice or Multi-speaker? (s/m) [default: m]: ").strip().lower() or 'm'

    if single_take_mode:
        director_mode = False
        if mode == 'm':
            print("Engaged: Full Story Single-Take with alternating multi-speaker turns.")
        else:
            print("Engaged: Full Story Single-Take with single speaker.")
    else:
        director_mode = input("Enable Director Mode to audition chunks? (y/n) [default: y]: ").strip().lower() != 'n'

    # Model engine selection
    if args.model:
        model_choice = args.model
    else:
        model_input = input("Select model (3.8-flash / 2.5-pro) [default: 3.8-flash]: ").strip().lower()
        model_choice = "2.5-pro" if ('2.5' in model_input or 'pro' in model_input) else "3.8-flash"

    if model_choice == "2.5-pro":
        model_name = DEFAULT_MODEL_25
        char_limit = CHAR_LIMIT_25
        print("Engaged: Gemini 2.5 Pro TTS (Fallback engine - Chunk limit: 3,000 chars)")
    else:
        model_name = DEFAULT_MODEL_38
        char_limit = CHAR_LIMIT_38_SINGLE_TAKE if single_take_mode else CHAR_LIMIT_38_CHUNKED
        print(f"Engaged: Gemini 3.8 Flash TTS (Primary flagship - Operational limit: {char_limit} chars)")

    if mode == 'm':
        spk1_name = input("Enter Speaker 1 Name [default: Narrator]: ").strip() or "Narrator"
        spk1_voice = input("Enter Speaker 1 Voice (e.g. Puck, Fenrir) [default: Puck]: ").strip() or "Puck"
        spk2_name = input("Enter Speaker 2 Name [default: Character]: ").strip() or "Character"
        spk2_voice = input("Enter Speaker 2 Voice (e.g. Kore, Zephyr) [default: Kore]: ").strip() or "Kore"
        voice_choice = f"Multi_{spk1_name}_{spk1_voice}_vs_{spk2_name}_{spk2_voice}"
    else:
        spk1_name = spk2_name = None
        spk1_voice = spk2_voice = None
        voice_choice = input("Enter Voice Name (e.g. Puck, Kore, Fenrir, Zephyr) [default: Puck]: ").strip() or "Puck"

    filename = args.script
    if not filename:
        filename = input("Enter manuscript path (.txt): ").strip()

    try:
        with open(filename, "r", encoding="utf-8") as file:
            manuscript_text = file.read().replace('\ufeff', '')
    except FileNotFoundError:
        print(f"Error: Manuscript file '{filename}' not found.")
        sys.exit(1)

    raw_chunks = manuscript_text.split("***")
    initial_chunks = [c.strip() for c in raw_chunks if c.strip()]
    total_chunks = len(initial_chunks)

    if total_chunks == 0:
        print("Error: No text found in manuscript file.")
        sys.exit(1)

    print(f"\nManuscript loaded: {total_chunks} vignette(s) identified via '***' breaks.")

    # Performance style direction
    style_prompt = args.style
    if not style_prompt:
        print("Tip: Specify volume/tone out-of-band (e.g. 'Clear natural projection, steady pacing').")
        style_prompt = input("Enter performance style prompt (or press Enter to skip): ").strip()

    url = f"{API_BASE_URL}/models/{model_name}:generateContent?key={secure_api_key}"

    # Cache setup
    base_name = os.path.splitext(os.path.basename(filename))[0]
    cache_dir = f".tts_cache_{base_name}_{voice_choice}"
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Active cache directory: {cache_dir}\n")

    if single_take_mode:
        master_output_filename = f"{base_name}_{voice_choice}_single_take.wav"
        print(f"[Single-Take] Packing {total_chunks} vignettes into a unified generation request...")

        full_request_parts = []
        for idx, chk in enumerate(initial_chunks, 1):
            if mode == 'm':
                expected_fallback = spk1_name if (idx % 2 != 0) else spk2_name
                chk_parts = build_request_parts(
                    chunk_text=chk,
                    mode='m',
                    spk1_name=spk1_name,
                    spk2_name=spk2_name,
                    style=style_prompt,
                    insert_lead_pause=(idx > 1),
                    default_speaker=expected_fallback
                )
            else:
                chk_parts = build_request_parts(
                    chunk_text=chk,
                    mode='s',
                    style=style_prompt,
                    insert_lead_pause=(idx > 1)
                )
            full_request_parts.extend(chk_parts)

        if not full_request_parts:
            print("Error: No speakable dialogue remaining after sanitizing manuscript.")
            sys.exit(1)

        # Pre-flight turn audit
        print("\n--- Audio Turn Audit ---")
        for i, p in enumerate(full_request_parts, 1):
            if mode == 'm':
                spk = p.get("speech_metadata", {}).get("speaker", "Unknown")
                voice = spk1_voice if spk.lower() == spk1_name.lower() else spk2_voice
            else:
                spk = "Narrator"
                voice = voice_choice
            preview = p.get("text", "")[:55].replace("\n", " ")
            print(f"  Turn {i:02d} | Speaker: {spk:<10} (Voice: {voice:<6}) | \"{preview}...\"")
        print("------------------------\n")

        if mode == 'm':
            speech_config = {
                "multiSpeakerVoiceConfig": {
                    "speakerVoiceConfigs": [
                        {"speaker": spk1_name, "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": spk1_voice}}},
                        {"speaker": spk2_name, "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": spk2_voice}}}
                    ]
                }
            }
        else:
            speech_config = {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {"voiceName": voice_choice}
                }
            }

        payload = {
            "contents": [{"role": "user", "parts": full_request_parts}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
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

        total_chars = sum(len(p.get("text", "")) for p in full_request_parts)
        print(f"Synthesizing master ({total_chars} spoken chars across {len(full_request_parts)} turns)...")

        response = send_tts_request(url, payload)
        if not response:
            sys.exit(1)

        resp_json = response.json()
        try:
            b64_data = resp_json['candidates'][0]['content']['parts'][0]['inlineData']['data']
            raw_audio = base64.b64decode(b64_data)
            save_audio_payload(raw_audio, master_output_filename)
            print(f"\n[+] Master Complete: {master_output_filename}")
            play_audio(master_output_filename)
            sys.exit(0)
        except (KeyError, IndexError) as err:
            print(f"[!] Error parsing API response payload: {err}")
            print(resp_json)
            sys.exit(1)

    index = 1
    while index <= total_chunks:
        chunk_wav_path = os.path.join(cache_dir, f"chunk_{index:03d}.wav")
        chunk_txt_path = os.path.join(cache_dir, f"chunk_{index:03d}.txt")
        chunk_orig_path = os.path.join(cache_dir, f"chunk_{index:03d}.orig.txt")

        # Live disk synchronization
        while True:
            try:
                with open(filename, "r", encoding="utf-8") as file:
                    live_text = file.read().replace('\ufeff', '')
            except FileNotFoundError:
                print(f"Error: Manuscript '{filename}' was moved or deleted.")
                sys.exit(1)

            live_chunks = [c.strip() for c in live_text.split("***") if c.strip()]
            if index > len(live_chunks):
                print(f"\n[!] Structural mismatch: Expected chunk {index}, but file only contains {len(live_chunks)} chunks.")
                input("    Restore the '***' delimiters on disk and press ENTER to re-read... ")
                continue

            raw_chunk = live_chunks[index - 1]
            char_count = len(clean_text_for_tts(raw_chunk))

            # Chunk size guard
            if char_count > char_limit:
                print(f"\n[!] Size Warning: Chunk {index} contains {char_count} chars (Configured limit: {char_limit}).")
                print("    [O]verride and generate anyway")
                print("    [E]dit chunk in editor (insert '***' to split on the fly)")
                print("    [R]eload manuscript after external save")
                action = input("Select action (o/e/r) [default: e]: ").strip().lower() or 'e'

                if action == 'e':
                    edited_block = edit_text_in_editor(raw_chunk)
                    if "***" in edited_block:
                        sub_chunks = [c.strip() for c in edited_block.split("***") if c.strip()]
                        if len(sub_chunks) > 1:
                            print(f"\n[+] Splitting chunk {index} into {len(sub_chunks)} sub-chunks on disk.")
                            live_chunks[index - 1 : index] = sub_chunks
                            with open(filename, "w", encoding="utf-8") as f:
                                f.write("\n\n***\n\n".join(live_chunks))
                            total_chunks = len(live_chunks)
                            raw_chunk = sub_chunks[0]
                    else:
                        live_chunks[index - 1] = edited_block.strip()
                        with open(filename, "w", encoding="utf-8") as f:
                            f.write("\n\n***\n\n".join(live_chunks))
                        raw_chunk = edited_block.strip()
                    break
                elif action == 'o':
                    break
                else:
                    continue
            else:
                break

        source_ingest_text = raw_chunk

        if os.path.exists(chunk_wav_path) and os.path.exists(chunk_txt_path):
            with open(chunk_txt_path, "r", encoding="utf-8") as f:
                cached_edited = normalize_text(f.read())
            cached_orig = ""
            if os.path.exists(chunk_orig_path):
                with open(chunk_orig_path, "r", encoding="utf-8") as f:
                    cached_orig = normalize_text(f.read())

            target_compare = normalize_text(source_ingest_text)
            if target_compare == cached_edited or (cached_orig and target_compare == cached_orig):
                print(f"Skipping chunk {index}/{total_chunks} - Verified take found in cache.")
                index += 1
                continue

        chunk_approved = False
        needs_generation = True

        while not chunk_approved:
            if needs_generation:
                active_fallback = spk1_name if (index % 2 != 0) else spk2_name
                request_parts = build_request_parts(
                    chunk_text=raw_chunk,
                    mode=mode,
                    spk1_name=spk1_name,
                    spk2_name=spk2_name,
                    style=style_prompt,
                    default_speaker=active_fallback
                )

                if not request_parts:
                    print(f"Notice: Chunk {index} contains no speakable dialogue. Skipping.")
                    chunk_approved = True
                    break

                if mode == 'm':
                    speech_config = {
                        "multiSpeakerVoiceConfig": {
                            "speakerVoiceConfigs": [
                                {"speaker": spk1_name, "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": spk1_voice}}},
                                {"speaker": spk2_name, "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": spk2_voice}}}
                            ]
                        }
                    }
                else:
                    speech_config = {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {"voiceName": voice_choice}
                        }
                    }

                payload = {
                    "contents": [{"role": "user", "parts": request_parts}],
                    "generationConfig": {
                        "responseModalities": ["AUDIO"],
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

                print(f"Synthesizing chunk {index}/{total_chunks} ({len(clean_text_for_tts(raw_chunk))} chars)...")
                response = send_tts_request(url, payload)
                if not response:
                    sys.exit(1)

                response_data = response.json()
                try:
                    b64_audio = response_data['candidates'][0]['content']['parts'][0]['inlineData']['data']
                    raw_audio_data = base64.b64decode(b64_audio)
                    save_audio_payload(raw_audio_data, chunk_wav_path)

                    with open(chunk_txt_path, "w", encoding="utf-8") as f:
                        f.write(raw_chunk)
                    with open(chunk_orig_path, "w", encoding="utf-8") as f:
                        f.write(source_ingest_text)

                    needs_generation = False
                except (KeyError, IndexError):
                    print(f"\n[!] Safety filter triggered or unexpected response on chunk {index}.")
                    print(response_data)
                    choice = input("    [E]dit text to resolve flag, or [A]bort? (e/a) [default: e]: ").strip().lower() or 'e'
                    if choice == 'e':
                        raw_chunk = edit_text_in_editor(raw_chunk)
                        needs_generation = True
                        continue
                    else:
                        print("Run aborted. Cached takes remain safe.")
                        sys.exit(1)

            if not director_mode:
                chunk_approved = True
            else:
                play_audio(chunk_wav_path)
                print(f"\n[ Review Take | Chunk {index}/{total_chunks} ]")
                choice = input("    [K]eep take\n    [R]e-roll delivery\n    [E]dit script\nSelect (k/r/e) [default: k]: ").strip().lower() or 'k'

                if choice == 'r':
                    print("    Discarding take and requesting fresh delivery...")
                    needs_generation = True
                    time.sleep(1)
                elif choice == 'e':
                    edited = edit_text_in_editor(raw_chunk)
                    if "***" in edited:
                        sub_chunks = [c.strip() for c in edited.split("***") if c.strip()]
                        if len(sub_chunks) > 1:
                            print(f"\n[+] Split Detected: Dividing chunk {index} into {len(sub_chunks)} sub-chunks on disk.")
                            live_chunks[index - 1 : index] = sub_chunks
                            with open(filename, "w", encoding="utf-8") as f:
                                f.write("\n\n***\n\n".join(live_chunks))
                            total_chunks = len(live_chunks)
                            raw_chunk = sub_chunks[0]
                        else:
                            raw_chunk = edited.strip()
                    else:
                        raw_chunk = edited.strip()
                        live_chunks[index - 1] = raw_chunk
                        with open(filename, "w", encoding="utf-8") as f:
                            f.write("\n\n***\n\n".join(live_chunks))
                    needs_generation = True
                else:
                    chunk_approved = True

        index += 1

    output_filename = f"{base_name}_{voice_choice}.wav"
    output_text_filename = f"{base_name}_revised.txt"
    pause_duration = args.pause
    print(f"\nAll {total_chunks} chunks approved. Concatenating master release files ({pause_duration}s pauses)...")

    # Read properties from first chunk to ensure precision stitching
    first_chunk_wav = os.path.join(cache_dir, "chunk_001.wav")
    with wave.open(first_chunk_wav, "rb") as first_wf:
        channels = first_wf.getnchannels()
        sampwidth = first_wf.getsampwidth()
        framerate = first_wf.getframerate()

    with wave.open(output_filename, "wb") as master_wf:
        master_wf.setnchannels(channels)
        master_wf.setsampwidth(sampwidth)
        master_wf.setframerate(framerate)

        for i in range(1, total_chunks + 1):
            chunk_wav_path = os.path.join(cache_dir, f"chunk_{i:03d}.wav")
            with wave.open(chunk_wav_path, "rb") as chunk_wf:
                master_wf.writeframes(chunk_wf.readframes(chunk_wf.getnframes()))

            if i < total_chunks:
                silence_frames = b"\x00" * int(framerate * sampwidth * channels * pause_duration)
                master_wf.writeframes(silence_frames)

    final_text_chunks = []
    for i in range(1, total_chunks + 1):
        chunk_txt_path = os.path.join(cache_dir, f"chunk_{i:03d}.txt")
        with open(chunk_txt_path, "r", encoding="utf-8") as f:
            final_text_chunks.append(f.read())

    with open(output_text_filename, "w", encoding="utf-8") as f:
        f.write("\n\n***\n\n".join(final_text_chunks))

    print(f"\n[+] Production Complete!")
    print(f"    Master WAV:     {output_filename}")
    print(f"    Revised Script: {output_text_filename}")

    cleanup = input(f"\nRetain chunk cache ({cache_dir}) for future re-edits? [Y/n]: ").strip().lower()
    if cleanup == 'n':
        shutil.rmtree(cache_dir)
        print("Working cache cleared.")
    else:
        print("Working cache preserved.")

if __name__ == "__main__":
    main()
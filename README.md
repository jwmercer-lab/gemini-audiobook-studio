# **Gemini Audiobook Production Studio**

A lightweight, zero-dependency Python production suite for narrating audiobooks, dramatic audio plays, and multi-speaker podcasts using **Google Gemini 3.8 Flash TTS** and **Gemini 2.5 Pro TTS**.

Designed for authors and audio producers, this tool handles turn-by-turn multi-speaker routing, out-of-band performance styling, in-flight live text editing, scene-break pause injection, and resumable session caching.

---

## **Key Features**

* **Dual Workflows:**
  * **[1] Chunked Director Mode:** Audition, approve, re-roll, or edit each scene/vignette individually as it generates.
  * **[2] Full Story Single-Take:** Synthesizes an entire multi-speaker manuscript in one continuous API call with automatic turn-taking.
* **Turn-Based Multi-Speaker Routing:** Automatically parses dialogue tags (e.g., `Narrator:`, `Sam:`) into structured `speech_metadata.speaker` turns. The engine speaks pure verbatim dialogue without reading character names aloud.
* **Stage Direction & Tag Sanitization:** Strips bracketed stage notes (`[SCENE TRANSITION: ...]`) and comments (`#`) so foley cues are never read aloud. Automatically converts invalid `<long pause>` tags into supported pause tokens.
* **Dual-Match Quota Caching:** Saves approved `.wav` takes alongside normalized script buffers. Re-running after a crash or network interruption resumes where you left off without burning API tokens.
* **In-Flight System Editor:** Launches your OS editor (`Notepad` on Windows, `TextEdit` on macOS, GUI/`nano` on Linux) whenever an edit or split is needed. Inserting `***` divides long passages into sub-chunks on the fly.
* **Zero Audio Dependencies:** Uses Python's native `wave` library to stitch 24 kHz 16-bit Mono master WAVs with configurable digital silence pauses. No FFmpeg, PyDub, or NumPy required.

---

## **Engine Models & Character Limits**

| Model Engine | Model ID | Recommended Chunk | Max Single-Take | Ideal Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Gemini 3.8 Flash TTS (Flagship)** | `gemini-3.8-flash-tts` | **4,500 chars** | **8,000 chars** | High emotional fidelity, dramatic pacing, multi-speaker dialogues |
| **Gemini 2.5 Pro TTS (Fallback)** | `gemini-2.5-pro-preview-tts` | **3,000 chars** | **4,000 chars** | Extended continuity, steady narrative prose |

---

## **Prebuilt Voice Roster**

| Voice | Gender / Quality | Best Character Archetype |
| :--- | :--- | :--- |
| **`Puck`** | Young / Upbeat / Mid-High | Anxious apprentice, young protagonist, brisk conversationalist |
| **`Kore`** | Delicate / Breathy / Clear | Poised heroine, porcelain elegance, fragile or intimate narration |
| **`Fenrir`** | Deep / Gravelly / Firm | Authoritative commander, grizzled detective, antagonist |
| **`Zephyr`** | Bright / Energetic | Youthful, inquisitive, fast-paced modern narrator |
| **`Charon`** | Informative / Steady | Classic documentary or non-fiction audiobook narrator |
| **`Aoede`** | Breezy / Warm | Conversational storyteller, light drama |

---

## **Installation**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/gemini-audiobook-studio.git
   cd gemini-audiobook-studio
   ```

2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Only requires Python 3.10+ and `requests`).*

3. **Set your Google Gemini API Key:**
   Get a free or paid API key from [Google AI Studio](https://aistudio.google.com/apikey).

   * **Linux/macOS:**
     ```bash
     export GEMINI_API_KEY="your-api-key-here"
     ```
   * **Windows (Command Prompt):**
     ```cmd
     set GEMINI_API_KEY="your-api-key-here"
     ```
   * **Windows (PowerShell):**
     ```powershell
     $env:GEMINI_API_KEY="your-api-key-here"
     ```

---

## **Manuscript Formatting Guide**

### 1. Scene Breaks (`***`)
Separate scenes, vignettes, or paragraphs using three asterisks on their own line:

```text
The heavy door swung shut against the rain. Marcus set his pack on the floorboards.

***

Across the counter, Sarah was already loading the ledger into her bag.
```

### 2. Multi-Speaker Dialogue
Prefix character lines with their configured name followed by a colon or dash. The engine strips the prefix and assigns the turn to the actor's designated voice:

```text
Narrator: The blue light burned against the frosted glass.

Sam: "Listen to me," he whispered, stepping between her and the counter.

Sarah: "I don't need your protection, Sam."
```

### 3. Stage Directions & Non-Verbal Tags
* **Stage Directions (Filtered):** Bracketed notes like `[SCENE TRANSITION: Bell rings]` or developer comments `# Note to self` are automatically stripped before sending to the TTS engine.
* **Inline Vocal Bursts:** Supported native tags include `<short pause>`, `<pause>`, and `<whisper>`.

---

## **Usage & CLI Commands**

### Interactive Mode (Default)
Run without flags to step through the guided setup:
```bash
python audiobook_studio.py
```

### Command Line Flags
You can also launch directly with arguments:

```bash
# Full Story Single-Take Pass
python audiobook_studio.py story.txt --workflow single-take --mode m --style "Dramatic narration, natural volume"

# Director Mode with a custom 2.0-second scene pause
python audiobook_studio.py story.txt --workflow chunked --mode m --pause 2.0
```

### CLI Flag Reference
* `script`: Path to your `.txt` manuscript.
* `--key`: Gemini API key (defaults to `$GEMINI_API_KEY`).
* `--workflow`: `chunked` (Director review) or `single-take` (full pass).
* `--mode`: `s` (Single voice) or `m` (Multi-speaker).
* `--model`: `3.8-flash` (default) or `2.5-pro`.
* `--style`: Out-of-band performance direction.
* `--pause`: Digital silence between scenes in seconds (default: `1.5`).

---

## **Director Mode Workflow**

When Chunked Director Mode is active, each audio file auditions automatically in your media player as soon as synthesis completes:

* **`[K]eep` (default):** Approves the take, commits it to the cache directory, and advances to the next scene.
* **`[R]e-roll`:** Discards the take and queries Gemini again for a fresh delivery and cadence.
* **`[E]dit`:** Spawns your operating system's native text editor. Adjust words, fix pronunciations, or add `***` to split an oversized chunk into smaller pieces.

---

## **Output Artifacts**

* **`{script}_{voices}.wav`**: Concatenated 24 kHz, 16-bit Mono master audio release.
* **`{script}_revised.txt`**: Stitched manuscript capturing all runtime edits and inline scene splits.
* **`.tts_cache_{script}_{voices}/`**: Session cache containing isolated takes (`chunk_001.wav`), active text cards (`chunk_001.txt`), and source references (`chunk_001.orig.txt`).

---

## **License**

Distributed under the MIT License. Open-source and free for commercial and non-commercial audiobook production.
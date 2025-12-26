# Gemini Audiobook Studio

A high-performance Python studio for generating long-form audiobooks using Google's Gemini 2.5 TTS models. This tool handles text chunking, parallel generation, quality control, and final assembly into a seamless audio file.

## Features

* **Parallel Generation:** Uses threaded workers to maximize throughput while respecting API rate limits (handling 429s automatically).

* **Project Management:** Automatically organizes output into `output/{ProjectName}/` folders, keeping your artifacts clean and separated.

* **Smart Resume & Text Verification:** If you stop and restart a project, the script doesn't just check if a file exists—it checks if the *text* matches.
  * **Safe Limit Adjustments:** You can change the character limit or edit your source text between runs. The script reads the `.txt` verification card for each existing chunk. If the text has shifted due to a new character limit, it automatically invalidates and regenerates that specific chunk while keeping the valid ones.

* **Interactive Budget Calculator:** Before generating, the script enters a planning loop. It calculates exactly how many requests your project requires based on your character limit and compares it to your daily API quota.
    * **Safety Margin:** It explicitly tells you how many "spare" requests you have for retries.
    * **Tuning:** You can adjust the character limit up or down in real-time to find the sweet spot between audio quality (shorter chunks) and quota efficiency (longer chunks).

* **Daily Batching:** specific feature for low-quota environments (like the 50 RPD limit on Pro). You can tell the script to process only a set number of chunks (e.g., "Run 20 chunks"). This allows you to spread a large project over several days without hitting quota errors.

* **Pause Handling (Scene Breaks):** Supports custom silence tags. Insert `<break time="2.0s" />` directly into your text file to create dramatic pauses or scene transitions (replacing traditional `***` breaks).

* **Smart Quality Control (QC):** Automatically flags audio defects using signal analysis:
  * **Dead Air:** Detects excessive silence.
  * **Hallucination Loops:** Flags clips that are impossibly long for the text provided.
  * **Metallic/Robotic Noise:** Uses Zero-Crossing Rate (ZCR) to detect "buzzy" audio artifacts.
  * **Monotone Voices:** Analyzes dynamic range to flag "flat" or bored-sounding generations.
  * **Voice Drift Sentry:** Uses Harmonic Product Spectrum (HPS) pitch detection to ensure the narrator stays in character.
    * **Male Mode:** Flags if pitch drifts too high (>175Hz), indicating a female/child hallucination.
    * **Female Mode:** Flags if pitch drifts too low (<155Hz), indicating a male hallucination.

* **Director Mode with Context:** An interactive review phase allowing you to audit, edit, and retry clips. Includes **Contextual Preview**, which plays the last 3 seconds of the *previous* chunk before the current one to ensure accent and tone continuity.

* **Auto-Stitching:** Automatically merges approved chunks into a single master `.wav` or `.mp3` file.

## Prerequisites

Before running the script, ensure you have the following:

1. **Python 3.10+**: The script relies on modern Python features.

2. **FFmpeg**: **Required** for `pydub` to stitch chunks into MP3 format. Without this, you will only get WAV files.

3. **Google Gemini API Key**: You need a valid key from [Google AI Studio](https://aistudio.google.com/) with access to the `gemini-2.5-flash` and `gemini-2.5-pro` models.

## Setup & Installation

### Linux (Ubuntu/Debian/Mint)

Since you are running this primarily on Linux, here is the fast track:

1. **Install System Dependencies:**

   ```bash
   sudo apt update
   sudo apt install python3 python3-pip ffmpeg git -y
   ```

2. **Clone the Repository:**

   ```bash
   git clone https://github.com/jwmercer-lab/gemini-audiobook-studio.git
   cd gemini-audiobook-studio
   ```

3. **Install Python Libraries:**

   ```bash
   pip3 install -r requirements.txt
   ```

### Windows

1. **Install Python:** Download and install Python 3.10+ from [python.org](https://www.python.org/downloads/). Ensure you check **"Add Python to PATH"** during installation.

2. **Install FFmpeg:**

   * Download a build from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/).

   * Extract the zip file.

   * Add the `bin` folder (e.g., `C:\ffmpeg\bin`) to your System Environment Variables -> Path.

3. **Install Python Libraries:**
   Open PowerShell or Command Prompt and run:

   ```powershell
   pip install -r requirements.txt
   ```

## Usage

1. Run the script:

   ```bash
   python3 audiobook_generator.py
   ```

2. Follow the prompts:

   * **Project Name:** Defines the output folder (e.g., output/`title`/).

   * **Resume/Overwrite:** If chunks exist in that folder, you can Resume to save time.

   * **Model & Voice:** Choose your settings.

3. **Budget Loop:** The script will show the default chunk limit (1500 for Flash, 2400 for Pro).

   * Press ENTER to calculate the plan.

   * Review the **Safety Margin** (spare requests).

   * If the margin is negative or too tight, enter a lower character limit to reduce the chunk count, or accept the risk.

4. **Batch Limit**

   * Press ENTER to run the whole project.

   *Or enter a number (e.g., `40`) to only run that many chunks. You can run the script again tomorrow to finish the rest.

### Director Mode Guide

If you enable Director Mode, the script will pause after generating all chunks to let you review them.

**The Workflow:**

1. The script plays the generated clip (preceded by 3 seconds of the *previous* clip for context).

2. You choose an action:

   * `[K]eep`: The clip is good. Move to the next one.

   * `[R]etry`: The clip sounded robotic or the accent slipped. The script immediately regenerates it.

   * `[E]dit Text`: The model stumbled on a specific word. You can edit the text in a temp file, and the script will regenerate the audio using your new text.

   * `[D]iscard`: Throw the whole project away (rarely used).

## Tuning Quality Control

If the script is being too strict (flagging good audio) or too lenient (missing robotic voices), you can adjust the thresholds in `audiobook_generator.py`:

* `QC_STRICT_SILENCE`: Max allowed silence in seconds (Default: 3.0s).

* **ZCR (Zero-Crossing Rate):** Detects hiss and metallic noise.
  * `0.20`: (Default) Tuned to catch "Flash Model" hiss and severe robotic glitches.
  * `0.18`: Stricter. Recommended if you hear faint hiss passing through.
  * `0.25`: Loose. Only catches extreme robotic failures.

* **RMS Std Dev (Dynamic Range):** Detects monotone voices.
  * `150`: (Default) Catches very flat, bored readings.
  * `200`: Stricter. Requires more emotional variance/drama to pass.

## Directory Structure

```plaintext
output/
└── MyAudiobook/
    ├── chunks/
    │   ├── chunk_0000.pcm
    │   ├── chunk_0001.pcm
    │   └── ...
    ├── temp_master.wav
    └── final_audiobook.mp3
```

## License

Distributed under the MIT License. See LICENSE for more information.
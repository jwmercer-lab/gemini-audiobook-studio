# **Gemini Audiobook Studio**

A lightweight, high-precision Python script for narrating stories and scripts using Google's Gemini TTS models. This tool handles hard scene breaks, multi-speaker dialogues, pre-flight payload validation, and rate limit survival to ensure you don't waste API quota on broken requests.

## **Features**

* **Multi-Speaker Support:** Natively supports two-character dialogue by mapping inline tags (e.g., Sarah: and Sam:) to distinct Gemini voices.
* **In-Flight System Editor:** Automatically launches your desktop text editor (xed, gedit, mousepad, kate on Linux; TextEdit on macOS; Notepad on Windows) whenever an edit is needed.
* **Dynamic Inline Splitting:** Adding `***` inside the editor splits an oversized chunk on the fly, updates the source text on disk, and recalculates the queue automatically.
* **Safety Filter Interception:** Catches tripped content filters mid-generation and opens an editor buffer so you can adjust phrasing and retry without aborting the session.
* **Dual-Match Caching:** Saves generated `.wav` files alongside both modified `.txt` and original `.orig.txt` records, using newline normalization to prevent redundant API calls.
* **Director Mode:** Audition chunks as they finish generating with immediate options to keep, re-roll, or open the text editor for real-time rewrites.
* **Dual Master Export:** Automatically stitches the full 24kHz master audio file and outputs an updated manuscript (`_revised.txt`) reflecting all runtime edits.
* **Zero Audio Dependencies:** Concatenates raw audio frames using Python's native `wave` library with no FFmpeg or third-party audio packages required.

## **Prerequisites**

1. **Python 3.10+**: Requires a modern Python environment.  
2. **Google Gemini API Key**: A valid REST API key from Google AI Studio.
3. **Supported Text Editor**:
   * **Linux:** Uses `xed`, `gedit`, `mousepad`, `kate`, or falls back to `$EDITOR` / `nano`.
   * **macOS:** Uses `TextEdit`.
   * **Windows:** Uses `Notepad`.

*Note: FFmpeg, Numpy, and Pydub are not required.*

## **Setup & Installation**

1. **Setup the Directory:**  
   Create a folder for your project and place `audiobook_studio.py` and your text files inside it.  
2. **Install Python Libraries:**  
   The script relies almost entirely on standard library modules. Install the single external dependency:  
   ```bash
   pip install -r requirements.txt
   ```

## **Text Formatting Rules**

**1. Scene Breaks**

Break your text into smaller chunks using three asterisks on their own line. Remember to stay under the character limit for your chosen model (2000 for Flash, 3000 for Pro).

The heavy door swung shut.

***

She walked down the hallway.

**2. Multi-Speaker Tagging**

If using the multi-speaker mode, every paragraph must be explicitly tagged with the character's name exactly as you input it during the script setup.

Sarah: The blue light burned Sarah’s eyes. 

Sam: "Listen to me,"  
Sarah: Sam yelled, swiping her volume down.

## **Usage**

Run the script from your terminal:

```bash
python audiobook_studio.py
```

The script will prompt you through the setup:

1. **API Key:** Paste your key securely.  
2. **Mode:** Select s (single) or m (multi-speaker).  
3. **Director Mode:** Enable to listen and approve chunks dynamically as they are generated.  
4. **Model:** Hit Enter to use the default 3.1-flash, or type pro to fall back to 2.5-pro.  
5. **Speaker Setup (Multi-mode):** Define the names found in your text and map them to Gemini voices.  
6. **File Name:** Enter the target .txt file.  
7. **Style Prompt:** Enter an atmospheric direction for the read, or press Enter to skip.

### **Director Mode & Live Editing**

When Director Mode is active, each audio chunk plays automatically as it completes:

* **[K]eep:** Commits the current audio take to cache and proceeds to the next chunk.
* **[R]etry:** Discards the take and immediately re-queries the model with identical text.
* **[E]dit:** Spawns your text editor to adjust phrasing, fix phonetics, or insert `***` delimiters to split long passages into smaller sub-chunks.

The script will run its pre-flight check and build your cache. If everything clears, it will transmit the chunks and output a single .wav file tagged with your chosen voices, prompting you to clean up the temporary files when finished.

## **Output Files**

* **`{base_name}_{voice_choice}.wav`**: The complete concatenated master audio file (24,000 Hz, 16-bit Mono).
* **`{base_name}_revised.txt`**: The final stitched manuscript containing every manual revision, phonetic adjustment, and scene split made during the run.
* **`.tts_cache_{base_name}_{voice_choice}/`**: Hidden working directory storing individual chunk audio files (`chunk_XXX.wav`), active text cards (`chunk_XXX.txt`), and raw original text (`chunk_XXX.orig.txt`).

## **License**

Distributed under the MIT License. Copyright (c) 2026 J.W. Mercer Lab.
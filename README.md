# Gemini Audiobook Studio

A Python-based studio for converting text files into full-length audiobooks using Google's Gemini Flash/Pro models.

## Features
* **Parallel Processing:** Uses concurrent workers to speed up generation (Threaded).
* **Smart Stitching:** Handles long texts by chunking them logically (3000 char limit).
* **Director Mode:** Infinite retry loop allows you to edit text, listen to previews, or regenerate specific chunks before finalizing.
* **Pause Handling:** Supports `<break time="2s"/>` tags for dramatic pacing (inserts true digital silence).
* **Auto-Recovery:** Detects "dead air" or silence from the API and automatically retries.

## Prerequisites
1.  **Python 3.8+**
2.  **Google Gemini API Key** (Get one from Google AI Studio)
3.  **FFmpeg** (Required for audio processing)

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/gemini-audiobook-studio.git
cd gemini-audiobook-studio
```

### 2. Install Dependencies

**For Linux (Debian/Ubuntu/Mint):**
```bash
sudo apt update
sudo apt install ffmpeg
pip install -r requirements.txt
```

**For Windows:**
1.  Download and install [FFmpeg](https://ffmpeg.org/download.html) (ensure it is added to your System PATH).
2.  Open PowerShell or Command Prompt.
3.  Run:
    ```powershell
    pip install -r requirements.txt
    ```
    
## Usage

1.  Place your text file (e.g., `my_book.txt`) inside the project folder.
2.  Run the script:
    ```bash
    python audiobook_studio.py
    ```
    3.  **Enter Inputs:**
    * Paste your Gemini API Key.
    * Enter the filename (e.g., `my_book.txt`).
    * Select Model (Flash for speed, Pro for quality).
    * Select Voice.

## Director Mode Tips
* **[R]etry:** Simply tries the generation again.
* **[E]dit:** Opens the text chunk in your default editor. Fix typos or adjust phrasing, save, and close to regenerate.
* **[L]isten:** (If available) plays the failed audio so you can decide if it's actually acceptable.

## License
Open Source.

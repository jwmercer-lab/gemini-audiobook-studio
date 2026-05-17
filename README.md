# **Gemini Audiobook Studio**

A lightweight, high-precision Python script for narrating stories and scripts using Google's Gemini TTS models. This tool handles hard scene breaks, multi-speaker dialogues, pre-flight payload validation, and rate limit survival to ensure you don't waste API quota on broken requests.

## **Features**

* **Multi-Speaker Support:** Natively supports two-character dialogue. Map inline tags (e.g., Sarah: and Sam:) to distinct Gemini voices for dynamic scene reads.  
* **Smart Pre-Flight Validation:** Splits your text along natural \*\*\* scene breaks. Before making a single API call, the script measures every chunk against the hard 3000-character limit and flags oversized blocks so you can cut them manually.  
* **Smart Caching & Resume:** The script saves individual .wav chunks and .txt verification cards to a temporary local directory. If you hit a hard failure, run the script again. It verifies the text logic and instantly skips over already-generated chunks.  
* **Rate Limit Survival:** Automatically catches 429 Too Many Requests errors from the API, initiates a cooldown, and retries without dropping the execution or losing data.  
* **Director Mode:** Review chunks as they generate. The script will pause, play the audio, and let you choose to keep the take or discard and re-roll it on the spot to catch hallucinated accents or flat reads.  
* **Variable Model Selection:** Toggle between gemini-2.5-pro-tts and gemini-3.1-flash-tts-preview.  
* **Atmospheric Prompts:** Inject a style prompt at runtime to dictate the baseline mood of the read.  
* **In-Memory Stitching:** No third-party audio dependencies like FFmpeg. The script stitches the local cache files using native Python modules and builds a single, continuous .wav file at the end.

## **Prerequisites**

1. **Python 3.10+**: Requires a modern Python environment.  
2. **Google Gemini API Key**: A valid REST API key from Google AI Studio.

*Note: FFmpeg, Numpy, and Pydub are no longer required.*

## **Setup & Installation**

1. **Setup the Directory:**  
   Create a folder for your project and place tts.py and your text files inside it.  
2. **Install Python Libraries:**  
   The script relies almost entirely on standard library modules. Install the single external dependency:  
   pip install \-r requirements.txt

## **Text Formatting Rules**

**1\. Scene Breaks**

Break your text into smaller chunks using three asterisks on their own line:

The heavy door swung shut.

\*\*\*

She walked down the hallway.

**2\. Multi-Speaker Tagging**

If using the multi-speaker mode, every paragraph must be explicitly tagged with the character's name exactly as you input it during the script setup.

Sarah: The blue light burned Sarah’s eyes. 

Sam: "Listen to me,"  
Sarah: Sam yelled, swiping her volume down.

## **Usage**

Run the script from your terminal:

python tts.py

The script will prompt you through the setup:

1. **API Key:** Paste your key securely.  
2. **Mode:** Select s (single) or m (multi-speaker).  
3. **Director Mode:** Enable to listen and approve chunks dynamically as they are generated.  
4. **Model:** Hit Enter to use 2.5-pro, or type 3.1-flash.  
5. **Speaker Setup (Multi-mode):** Define the names found in your text and map them to Gemini voices.  
6. **File Name:** Enter the target .txt file.  
7. **Style Prompt:** Enter an atmospheric direction for the read, or press Enter to skip.

The script will run its pre-flight check and build your cache. If everything clears, it will transmit the chunks and output a single .wav file tagged with your chosen voices, prompting you to clean up the temporary files when finished.
# Audio-Manuscript Aligner

This tool synchronizes an Input Text (Manuscript) with an Audio file using FFmpeg and OpenAI Whisper.

## Purpose
The tool maps existing text content to audio timestamps. It does not generate new text; instead, it uses AI to identify the timing of speech and applies the provided text to those time intervals.

## Functional Overview
* **Text Retention**: Uses the provided input text as-is.
* **Interval Detection**: Uses Whisper to identify speech positions.
* **Format Handling**: Automatically converts non-WAV inputs to temporary 16kHz WAV files.
* **Output Formats**: Produces individual .txt files per segment and a consolidated .srt file.

## Prerequisites
* Python 3.8 - 3.11
* FFmpeg (External executable)
* Python Libraries: `openai-whisper`, `torch`, `setuptools-rust`

## Usage
```bash
python srt_gen.py <audio_file> <manuscript_file>
```

## Processing Steps
1. **Silence Analysis**: FFmpeg identifies silence/speech boundaries.
2. **Audio Decoding**: Non-WAV files are converted to a temporary WAV format.
3. **Speech-to-Text Analysis**: Whisper transcribes segments for positioning.
4. **Fuzzy Matching**: Matches AI transcription with the input text using Python's standard libraries.
5. **File Generation**: Saves result files.

## License
MIT

## Credits
This tool utilizes the following external components:
* [OpenAI Whisper](https://github.com/openai/whisper) (MIT License)
* [FFmpeg](https://ffmpeg.org/) (LGPL/GPL License)

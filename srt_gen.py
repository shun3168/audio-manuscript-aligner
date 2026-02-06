import os
import subprocess
import re
import whisper
import io
import sys
import tempfile
from difflib import SequenceMatcher
from datetime import timedelta

def format_srt_time(seconds):
    """
    Converts seconds into SRT format: HH:MM:SS,mmm
    Example: 61.5 -> 00:01:01,500
    """
    td = timedelta(seconds=seconds)
    total_sec = int(td.total_seconds())
    msec = int(td.microseconds / 1000)
    return f"{total_sec//3600:02}:{(total_sec%3600)//60:02}:{total_sec%60:02},{msec:03}"

def get_refined_split_pos(curr_heard, next_heard, script_segment, base_limit, extended_limit, long_wav_sec, dur):
    """
    Calculates where to cut the script text to match the audio.
    It compares Whisper's transcript with the master script to find the best cut point.
    """
    # Use the last 2 words of the current clip and the first 2 of the next clip as anchors
    words_next = next_heard.split()[:2]
    words_curr = curr_heard.split()[-2:]
    t_next = " ".join(words_next)
    t_curr = " ".join(words_curr)
    
    # Increase search range for long clips to avoid cutting the text too early
    limit = extended_limit if dur >= long_wav_sec else base_limit
    segment = script_segment[:limit]
    
    best_pos, max_score = 0, -1
    
    # Sliding window: check every character position for the best match
    for i in range(len(segment) + 1):
        # Match current tail
        s_left = SequenceMatcher(None, t_curr, segment[max(0, i-len(t_curr)):i].strip()).ratio() if t_curr else 0
        # Match next head
        s_right = SequenceMatcher(None, t_next, segment[i:i+len(t_next)].strip()).ratio() if t_next else 0
        
        # Combined match score (0.0 to 1.0)
        score = (s_left + s_right) / 2
        if score > max_score:
            max_score, best_pos = score, i
            
    return best_pos, max_score

def run():
    # --- 1. Input Check ---
    if len(sys.argv) < 3:
        print("USAGE: python srt_gen.py <audio_file> <script_file>")
        sys.exit(1)

    audio_in = sys.argv[1]
    script_arg = sys.argv[2]

    if not os.path.exists(audio_in):
        print(f"ERROR: Audio file '{audio_in}' not found.")
        sys.exit(1)

    # Check for script file with or without .txt
    script_in = script_arg
    if not os.path.exists(script_in):
        if os.path.exists(script_arg + ".txt"):
            script_in = script_arg + ".txt"
        else:
            print(f"ERROR: Script '{script_arg}' not found.")
            sys.exit(1)

    # --- 2. File & Folder Naming ---
    audio_name = os.path.splitext(os.path.basename(audio_in))[0]
    script_name = os.path.splitext(os.path.basename(script_in))[0]
    output_dir = script_name
    output_srt = audio_name + ".srt"

    # --- 3. Internal Format Conversion ---
    # If the input isn't a WAV, convert it to a temporary 16kHz WAV.
    # This is required for precise timing and Whisper compatibility.
    current_audio_workfile = audio_in
    temp_wav_file = None
    if not audio_in.lower().endswith(".wav"):
        print(f"NOTICE: Converting {audio_in} to a temporary WAV for precise processing...")
        temp_wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        # Force 16kHz Mono WAV (best for Whisper/FFmpeg analysis)
        conv_cmd = ["ffmpeg", "-y", "-i", audio_in, "-ac", "1", "-ar", "16000", temp_wav_file]
        subprocess.run(conv_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        current_audio_workfile = temp_wav_file

    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(script_in, "r", encoding="utf-8") as f:
            master_script = " ".join(f.read().split())

        # --- 4. Detect Speech Intervals ---
        print(f"STEP 1: Scanning {audio_in} for speech and silence...")
        log_cmd = ["ffmpeg", "-i", current_audio_workfile, "-af", "silencedetect=noise=-30dB:d=0.5", "-f", "null", "-"]
        result = subprocess.run(log_cmd, stderr=subprocess.PIPE, text=True, encoding="utf-8")
        
        silence_starts = re.findall(r"silence_start: ([\d\.]+)", result.stderr)
        silence_ends = re.findall(r"silence_end: ([\d\.]+)", result.stderr)

        # Create list of audio segments (start time, end time, duration)
        segments = []
        for i in range(len(silence_ends)):
            start = float(silence_ends[i])
            end = float(silence_starts[i+1]) if i + 1 < len(silence_starts) else start + 5.0
            if end - start > 0.1:
                segments.append({"start": start, "end": end, "dur": end - start})

        print(f"Total segments found: {len(segments)}")
        
        ai_model = None
        remaining_script = master_script
        srt_data = []

        # --- 5. Text-to-Audio Alignment Loop ---
        for i, seg in enumerate(segments):
            timestamp_ms = int(seg['start'] * 1000)
            txt_file_path = os.path.join(output_dir, f"{timestamp_ms:09d}.txt")
            
            # Skip if file exists (Resume feature)
            if os.path.exists(txt_file_path):
                with open(txt_file_path, "r", encoding="utf-8") as f:
                    saved_text = f.read().strip()
                print(f"[{i+1:03d}] SKIP: {os.path.basename(txt_file_path)} already exists.")
                # Move script pointer forward
                find_idx = remaining_script.find(saved_text)
                if find_idx != -1: remaining_script = remaining_script[find_idx + len(saved_text):].strip()
            else:
                if ai_model is None:
                    print("STEP 2: Loading Whisper AI...")
                    ai_model = whisper.load_model("turbo")

                # Extract audio segment and send to AI via memory pipe
                extract_cmd = ["ffmpeg", "-ss", str(seg['start']), "-t", str(seg['dur']), "-i", current_audio_workfile, "-f", "wav", "-ar", "16000", "-ac", "1", "pipe:1"]
                audio_buffer = subprocess.run(extract_cmd, capture_output=True).stdout
                transcript_curr = ai_model.transcribe(io.BytesIO(audio_buffer))["text"].strip()

                # Preview next segment to find the exact cut-off point
                if i + 1 < len(segments):
                    next_seg = segments[i+1]
                    next_cmd = ["ffmpeg", "-ss", str(next_seg['start']), "-t", str(next_seg['dur']), "-i", current_audio_workfile, "-f", "wav", "-ar", "16000", "-ac", "1", "pipe:1"]
                    next_buffer = subprocess.run(next_cmd, capture_output=True).stdout
                    transcript_next = ai_model.transcribe(io.BytesIO(next_buffer))["text"].strip()
                    
                    split_idx, match_score = get_refined_split_pos(transcript_curr, transcript_next, remaining_script, 45, 400, 5.5, seg['dur'])
                else:
                    split_idx = len(remaining_script)
                    match_score = 1.0

                final_segment_text = remaining_script[:split_idx].strip()
                
                with open(txt_file_path, "w", encoding="utf-8") as f:
                    f.write(final_segment_text)
                
                remaining_script = remaining_script[split_idx:].strip()
                print(f"[{i+1:03d}] SAVED: {os.path.basename(txt_file_path)} (Score: {match_score:.2f})")

            # Store for SRT generation
            start_time_srt = format_srt_time(seg['start'])
            end_time_srt = format_srt_time(seg['end'])
            srt_data.append(f"{i+1}\n{start_time_srt} --> {end_time_srt}\n{final_segment_text}\n")

        # --- 6. Save Final SRT ---
        with open(output_srt, "w", encoding="utf-8") as f:
            f.write("\n".join(srt_data))
        print(f"\nFINISH:\n- Text fragments: {output_dir}/\n- Subtitles: {output_srt}")

    finally:
        # --- 7. Clean up ---
        if temp_wav_file and os.path.exists(temp_wav_file):
            os.remove(temp_wav_file)
            print("CLEANUP: Temporary audio file deleted.")

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\nAborted by user.")
        sys.exit(0)
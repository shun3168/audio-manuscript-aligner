import os
import subprocess
import re
import whisper
import io
import sys
import tempfile
from difflib import SequenceMatcher
from datetime import timedelta

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

# Threshold for similarity score (0.0 to 1.0).
# If the average match of the left and right anchors is below this, 
# we consider the alignment unreliable for tail-correction.
SCORE_THRESHOLD = 0.5

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def format_srt_time(seconds):
    """
    PURPOSE: Converts a raw float (seconds) into the industry-standard SRT time string.
    LOGIC: Uses Python's timedelta to extract hours, minutes, and seconds,
           then manually calculates milliseconds for the ,mmm suffix.
    """
    td = timedelta(seconds=seconds)
    total_sec = int(td.total_seconds())
    msec = int(td.microseconds / 1000)
    # Output Format: HH:MM:SS,mmm (e.g., 00:05:22,450)
    return f"{total_sec//3600:02}:{(total_sec%3600)//60:02}:{total_sec%60:02},{msec:03}"

def get_refined_split_pos(curr_heard, next_heard, script_segment, base_limit, extended_limit, long_wav_sec, dur):
    """
    ALGORITHM: Sliding Window Phonetic Alignment.
    
    PURPOSE: To find the exact character index in the script where one audio clip 
             ends and the next begins.
             
    LOGIC:
    1. Defines two 'anchors': 
       - Left Anchor: The last 2 words Whisper heard in the current segment.
       - Right Anchor: The first 2 words Whisper heard in the next segment.
    2. Slides through the script character-by-character.
    3. At each index 'i', it calculates the SequenceMatcher ratio for both anchors.
    4. Returns the index 'i' that yields the highest average similarity score.
    """
    # Extract anchors from transcribed text
    words_next = next_heard.split()[:2]
    words_curr = curr_heard.split()[-2:]
    t_next = " ".join(words_next)
    t_curr = " ".join(words_curr)
    
    # Range is dynamically adjusted. Longer audio needs a larger 'look-ahead' 
    # in the script to find its end point.
    limit = extended_limit if dur >= long_wav_sec else base_limit
    segment = script_segment[:limit]
    
    best_pos, max_score = 0, -1
    
    # Brute-force search for the best split point within the segment
    for i in range(len(segment) + 1):
        # s_left matches the text leading up to 'i'
        s_left = SequenceMatcher(None, t_curr, segment[max(0, i-len(t_curr)):i].strip()).ratio() if t_curr else 0
        # s_right matches the text starting from 'i'
        s_right = SequenceMatcher(None, t_next, segment[i:i+len(t_next)].strip()).ratio() if t_next else 0
        
        # Combined arithmetic mean of anchor scores
        score = (s_left + s_right) / 2
        if score > max_score:
            max_score, best_pos = score, i
            
    return best_pos, max_score

# ==============================================================================
# MAIN EXECUTION ENGINE
# ==============================================================================

def run():
    # --- STEP 1: CLI ARGUMENT VALIDATION ---
    # Ensures the user provided the necessary input files.
    if len(sys.argv) < 3:
        print("USAGE: python srt_gen.py <audio_file> <script_file>")
        sys.exit(1)

    audio_in = sys.argv[1]
    script_arg = sys.argv[2]

    # File existence check to prevent early crashes
    if not os.path.exists(audio_in):
        print(f"ERROR: Audio file '{audio_in}' not found.")
        sys.exit(1)

    # Resolution logic: If 'myscript' is passed, try to find 'myscript.txt'
    script_in = script_arg
    if not os.path.exists(script_in):
        if os.path.exists(script_arg + ".txt"):
            script_in = script_arg + ".txt"
        else:
            print(f"ERROR: Script '{script_arg}' not found.")
            sys.exit(1)

    # Output paths configuration
    audio_name = os.path.splitext(os.path.basename(audio_in))[0]
    script_name = os.path.splitext(os.path.basename(script_in))[0]
    output_dir = script_name # Directory for individual .txt fragments
    output_srt = audio_name + ".srt" # Final subtitle file

    # --- STEP 2: AUDIO NORMALIZATION (FFMPEG) ---
    # Whisper AI is optimized for 16kHz Mono. Silence detection also requires 
    # consistent audio specs to work reliably across different source formats.
    current_audio_workfile = audio_in
    temp_wav_file = None
    if not audio_in.lower().endswith(".wav"):
        print(f"NOTICE: Creating normalized 16kHz Mono WAV workfile...")
        fd, temp_wav_file = tempfile.mkstemp(suffix=".wav")
        os.close(fd) 
        # Standardize: -ac 1 (Mono), -ar 16000 (16kHz)
        conv_cmd = ["ffmpeg", "-y", "-i", audio_in, "-ac", "1", "-ar", "16000", temp_wav_file]
        subprocess.run(conv_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        current_audio_workfile = temp_wav_file

    try:
        # Create output directory for caching results
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # CRITICAL DESIGN CHOICE: SCRIPT NORMALIZATION
        # To ensure perfect find() operations, we convert the entire script into 
        # a single string where every newline or multiple space is EXACTLY one space.
        # This is the "Master Truth" for all character index calculations.
        with open(script_in, "r", encoding="utf-8") as f:
            master_script = " ".join(f.read().split())

        # --- STEP 3: SILENCE-BASED AUDIO SEGMENTATION ---
        print(f"STEP 1: Analyzing audio for speech intervals...")
        # FFmpeg silencedetect detects gaps. We use a -30dB threshold.
        log_cmd = ["ffmpeg", "-i", current_audio_workfile, "-af", "silencedetect=noise=-30dB:d=0.5", "-f", "null", "-"]
        # errors="replace" prevents Windows-specific encoding crashes during log parsing.
        result = subprocess.run(log_cmd, stderr=subprocess.PIPE, text=True, errors="replace")
        
        # Regex extraction of timestamps
        silence_starts = re.findall(r"silence_start: ([\d\.]+)", result.stderr)
        silence_ends = re.findall(r"silence_end: ([\d\.]+)", result.stderr)

        # Create segment metadata (start, end, duration)
        segments = []
        for i in range(len(silence_ends)):
            start = float(silence_ends[i])
            # If no next silence, assume the segment lasts 5 seconds or until EOF
            end = float(silence_starts[i+1]) if i + 1 < len(silence_starts) else start + 5.0
            if end - start > 0.1: # Discard noise artifacts
                segments.append({"start": start, "end": end, "dur": end - start})

        total_count = len(segments)
        print(f"Total speech segments detected: {total_count}")
        
        ai_model = None
        remaining_script = master_script
        srt_data = []

        # --- STEP 4: TRANSCRIPTION & SCRIPT ALIGNMENT LOOP ---
        for i, seg in enumerate(segments):
            curr_num = i + 1
            timestamp_ms = int(seg['start'] * 1000)
            txt_filename = f"{timestamp_ms:09d}.txt"
            txt_file_path = os.path.join(output_dir, txt_filename)
            
            final_segment_text = ""
            match_score = 1.0 # Default score for resume/final

            # RESUME FEATURE: Perfect synchronization with previous runs.
            if os.path.exists(txt_file_path):
                with open(txt_file_path, "r", encoding="utf-8") as f:
                    # We read the file content as-is.
                    final_segment_text = f.read()
                
                # SEARCH LOGIC: Because we save normalized raw slices, 
                # find() will perform a 100% binary-accurate search in master_script.
                find_idx = remaining_script.find(final_segment_text)
                
                if find_idx != -1: 
                    # Advance the script pointer by the exact length of the found fragment.
                    remaining_script = remaining_script[find_idx + len(final_segment_text):].strip()
                else:
                    # Error indicates that either the master script or the fragment was modified.
                    print(f"ERROR: Synchronization lost at segment {curr_num}!")
            else:
                # LAZY LOAD: Initialize the AI model only when a new segment needs processing.
                if ai_model is None:
                    print("STEP 2: Initializing OpenAI Whisper AI (Turbo)...")
                    ai_model = whisper.load_model("turbo")

                # EXTRACTION: Write the specific audio segment to a temporary WAV.
                # This ensures the AI receives a clean, single-segment file.
                fd1, tmp_curr = tempfile.mkstemp(suffix=".wav")
                os.close(fd1)
                subprocess.run(["ffmpeg", "-y", "-ss", str(seg['start']), "-t", str(seg['dur']), "-i", current_audio_workfile, "-ar", "16000", "-ac", "1", tmp_curr], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # WHISPER TRANSCRIPTION: AI hears the audio and converts it to text.
                transcript_curr = ai_model.transcribe(tmp_curr)["text"].strip()
                os.remove(tmp_curr) 

                # ALIGNMENT: Calculate where the text ends by looking at the next segment.
                if curr_num < total_count:
                    next_seg = segments[i+1]
                    fd2, tmp_next = tempfile.mkstemp(suffix=".wav")
                    os.close(fd2)
                    subprocess.run(["ffmpeg", "-y", "-ss", str(next_seg['start']), "-t", str(next_seg['dur']), "-i", current_audio_workfile, "-ar", "16000", "-ac", "1", tmp_next], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    transcript_next = ai_model.transcribe(tmp_next)["text"].strip()
                    os.remove(tmp_next) 
                    
                    # PHASE A: Basic Sliding Window Alignment.
                    split_idx, match_score = get_refined_split_pos(transcript_curr, transcript_next, remaining_script, 45, 400, 5.5, seg['dur'])
                    
                    # PHASE B: TAIL-ANCHOR CORRECTION (The "origtxt" logic).
                    # Purpose: Whisper often detects words that might be slightly beyond the 
                    # mathematical cut point. This forces the cut point to extend until 
                    # it covers the actual words Whisper heard.
                    initial_slice = remaining_script[:split_idx].strip()
                    t_curr_end = " ".join(transcript_curr.split()[-2:]) 
                    
                    if match_score >= SCORE_THRESHOLD and t_curr_end:
                        # If the expected tail words are not in the current slice:
                        if t_curr_end not in initial_slice:
                            # Search slightly ahead in the master script for the tail words.
                            search_limit = 400 if seg['dur'] >= 5.5 else 45
                            found_pos = remaining_script[:search_limit].rfind(t_curr_end)
                            if found_pos != -1:
                                new_end_pos = found_pos + len(t_curr_end)
                                # Only update if the correction actually moves the pointer forward.
                                if new_end_pos > split_idx:
                                    split_idx = new_end_pos 
                else:
                    # FINAL SEGMENT: Absorb all remaining text in the master script.
                    split_idx = len(remaining_script)
                    match_score = 1.0

                # PERSISTENCE LOGIC: "Perfect Sync" Extraction.
                # We slice the EXACT normalized text from the master_script.
                final_segment_text = remaining_script[:split_idx].strip()
                
                # BINARY PRESERVATION: We save without extra formatting or newlines.
                # newline='' is essential to prevent OS-level translation of \n.
                with open(txt_file_path, "w", encoding="utf-8", newline='') as f:
                    f.write(final_segment_text)
                
                # Update the master pointer for the next iteration.
                remaining_script = remaining_script[split_idx:].strip()

            # --- PROGRESS VISUALIZATION ---
            # We replace newlines for the terminal preview only.
            preview = final_segment_text.replace("\n", " ")
            preview = (preview[:37] + "...") if len(preview) > 40 else preview
            print(f"[{curr_num:04d}/{total_count:04d}] (Score: {match_score:.2f}) {txt_filename} -> {preview}")

            # Append to internal list for final SRT assembly.
            start_time_srt = format_srt_time(seg['start'])
            end_time_srt = format_srt_time(seg['end'])
            srt_data.append(f"{curr_num}\n{start_time_srt} --> {end_time_srt}\n{final_segment_text}\n")

        # --- STEP 5: FINAL SRT ASSEMBLY & CLEANUP ---
        with open(output_srt, "w", encoding="utf-8") as f:
            f.write("\n".join(srt_data))
        print(f"\nFINISH:\n- All text fragments saved in: {output_dir}/\n- Master Subtitles created: {output_srt}")

    finally:
        # Crucial: Always remove the temporary normalized audio file.
        if temp_wav_file and os.path.exists(temp_wav_file):
            os.remove(temp_wav_file)
            print("CLEANUP: Process workfile removed.")

if __name__ == "__main__":
    # Provides KeyboardInterrupt (Ctrl+C) handling for clean exits.
    try:
        run()
    except KeyboardInterrupt:
        print("\nAborted by user.")
        sys.exit(0)

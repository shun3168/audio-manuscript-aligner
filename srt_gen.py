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
# CONFIGURATION: ALIGNMENT AND SYNCHRONIZATION THRESHOLDS
# ==============================================================================

# Minimum confidence score (0.0 to 1.0) required to accept a script alignment.
# If the similarity ratio is below 0.5, the script pointer is held constant
# to prevent incorrect text from being consumed and misaligning future segments.
SCORE_THRESHOLD = 0.5

# ==============================================================================
# UTILITY: SRT SPECIFICATION COMPLIANT FORMATTING
# ==============================================================================

def format_srt_time(seconds):
    """
    Converts float-based seconds into the strict SRT timestamp format: HH:MM:SS,mmm.
    SRT standards specifically require a comma (,) as the millisecond separator.
    """
    td = timedelta(seconds=seconds)
    total_sec = int(td.total_seconds())
    msec = int(td.microseconds / 1000)
    # Returns formatted string: e.g., 00:01:23,456
    return f"{total_sec//3600:02}:{(total_sec%3600)//60:02}:{total_sec%60:02},{msec:03}"

# ==============================================================================
# CORE ALGORITHM: SLIDING WINDOW ANCHOR MATCHING
# ==============================================================================

def get_refined_split_pos(curr_heard, next_heard, script_segment, base_limit, extended_limit, long_wav_sec, dur):
    """
    Determines the mathematically optimal character index to split the master script.
    
    1. ANCHORS: Extracts trailing words of current segment and leading words of 
       the next segment to serve as phonetic markers for the boundary.
    2. DYNAMIC LOOKAHEAD: Search range is adjusted based on audio duration
       to compensate for potential accumulation of timing drift.
    3. RATIO CALCULATION: Compares anchors against every possible position 'i'
       to find where the combined similarity ratio is maximized.
    """
    # Extract phonetic anchors for boundary verification
    words_next = next_heard.split()[:2]
    words_curr = curr_heard.split()[-2:]
    t_next = " ".join(words_next)
    t_curr = " ".join(words_curr)
    
    # Select search limit based on clip duration (Longer clips need more lookahead)
    limit = extended_limit if dur >= long_wav_sec else base_limit
    segment = script_segment[:limit]
    
    best_pos, max_score = 0, -1
    # Iterate through segment to find the point with highest phonetic similarity
    for i in range(len(segment) + 1):
        # s_left verifies the exit boundary / s_right verifies the entry boundary
        s_left = SequenceMatcher(None, t_curr, segment[max(0, i-len(t_curr)):i].strip()).ratio() if t_curr else 0
        s_right = SequenceMatcher(None, t_next, segment[i:i+len(t_next)].strip()).ratio() if t_next else 0
        
        # Calculate arithmetic mean to ensure both anchors are satisfied
        score = (s_left + s_right) / 2
        if score > max_score:
            max_score, best_pos = score, i
            
    return best_pos, max_score

# ==============================================================================
# MAIN PROCESSING ENGINE
# ==============================================================================

def run():
    # --- STEP 1: CLI ARGUMENT AND FILE PATH VALIDATION ---
    # Validates input availability before allocating heavy system resources for AI.
    if len(sys.argv) < 3:
        print("USAGE: python srt_gen.py <audio_file> <script_file>")
        sys.exit(1)

    audio_in = sys.argv[1]
    script_arg = sys.argv[2]

    # Verify existence of source audio file
    if not os.path.exists(audio_in):
        print(f"ERROR: Audio file '{audio_in}' not found.")
        sys.exit(1)

    # Resolution logic for script filename (handles extension-less input)
    script_in = script_arg
    if not os.path.exists(script_in):
        if os.path.exists(script_arg + ".txt"):
            script_in = script_arg + ".txt"
        else:
            print(f"ERROR: Script '{script_arg}' not found.")
            sys.exit(1)

    # Setup naming conventions for output directory and final SRT file
    audio_name = os.path.splitext(os.path.basename(audio_in))[0]
    script_name = os.path.splitext(os.path.basename(script_in))[0]
    output_dir = script_name
    output_srt = audio_name + ".srt"

    # --- STEP 2: AUDIO SIGNAL NORMALIZATION ---
    # Converts input to 16kHz Mono WAV. This specific format is mandatory for 
    # ensuring maximum accuracy in both Whisper AI and FFmpeg silence filters.
    current_audio_workfile = audio_in
    temp_wav_file = None
    if not audio_in.lower().endswith(".wav"):
        print(f"NOTICE: Normalizing audio to 16kHz Mono WAV workfile...")
        fd, temp_wav_file = tempfile.mkstemp(suffix=".wav")
        os.close(fd) 
        # ac 1 = Mono, ar 16000 = 16kHz sample rate
        conv_cmd = ["ffmpeg", "-y", "-i", audio_in, "-ac", "1", "-ar", "16000", temp_wav_file]
        subprocess.run(conv_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        current_audio_workfile = temp_wav_file

    try:
        # Create persistent storage for text fragments
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # SCRIPT NORMALIZATION: Collapses all erratic whitespaces and newlines.
        # This creates a reliable character-based coordinate system for alignment.
        with open(script_in, "r", encoding="utf-8") as f:
            master_script = " ".join(f.read().split())

        # --- STEP 3: VAD (VOICE ACTIVITY DETECTION) ---
        # Detects speech boundaries using decibel thresholds to isolate sentences.
        print(f"STEP 1: Analyzing audio for speech boundaries via FFmpeg...")
        
        # FIXED: Quotes correctly handled without backslash escape errors.
        log_cmd = ["ffmpeg", "-i", current_audio_workfile, "-af", "silencedetect=noise=-30dB:d=0.5", "-f", "null", "-"]
        # errors='replace' used to prevent crash on OS-specific log character encoding.
        result = subprocess.run(log_cmd, stderr=subprocess.PIPE, text=True, errors="replace")
        
        # Regex patterns to extract silence start/end timestamps from FFmpeg logs
        silence_starts = re.findall(r"silence_start: ([\d\.]+)", result.stderr)
        silence_ends = re.findall(r"silence_end: ([\d\.]+)", result.stderr)

        # Iterate through gaps between silence to identify active speech segments.
        segments = []
        for i in range(len(silence_ends)):
            start = float(silence_ends[i])
            # Default to 5s or EOF for the final detected segment.
            end = float(silence_starts[i+1]) if i + 1 < len(silence_starts) else start + 5.0
            if end - start > 0.1: # Skip micro-noises to prevent false subtitles.
                segments.append({"start": start, "end": end, "dur": end - start})

        total_count = len(segments)
        print(f"Total speech segments detected: {total_count}")
        
        ai_model = None
        remaining_script = master_script
        srt_data = []

        # --- STEP 4: TRANSCRIPTION AND ALIGNMENT LOOP ---
        # The core processing loop that maps each audio segment to a script fragment.
        for i, seg in enumerate(segments):
            curr_num = i + 1
            # Generate a unique timestamp-based filename for segment persistence.
            timestamp_ms = int(seg['start'] * 1000)
            txt_filename = f"{timestamp_ms:09d}.txt"
            txt_file_path = os.path.join(output_dir, txt_filename)
            
            final_segment_text = ""
            match_score = 1.0 # Default confidence for resumed data

            # STATE RECOVERY: Check for pre-existing fragments to support pause/resume.
            if os.path.exists(txt_file_path):
                with open(txt_file_path, "r", encoding="utf-8") as f:
                    final_segment_text = f.read()
                
                # If content exists, sync the global script pointer.
                if final_segment_text:
                    # Locate exact character sequence in remaining script for perfect sync.
                    find_idx = remaining_script.find(final_segment_text)
                    if find_idx != -1: 
                        # Update the pointer to point to the next available script text.
                        remaining_script = remaining_script[find_idx + len(final_segment_text):].strip()
                    else:
                        print(f"WARNING: Sync mismatch at segment {curr_num}. Pointer reset required.")
            else:
                # LAZY LOADING: Initialize Whisper model only upon encountering new segments.
                if ai_model is None:
                    print("STEP 2: Initializing OpenAI Whisper AI (Turbo model)...")
                    ai_model = whisper.load_model("turbo")

                # EXTRACTION: Write isolated segment to temporary file for AI input.
                fd1, tmp_curr = tempfile.mkstemp(suffix=".wav")
                os.close(fd1)
                subprocess.run(["ffmpeg", "-y", "-ss", str(seg['start']), "-t", str(seg['dur']), "-i", current_audio_workfile, "-ar", "16000", "-ac", "1", tmp_curr], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # Convert speech to raw text using Whisper phonetic processing.
                transcript_curr = ai_model.transcribe(tmp_curr)["text"].strip()
                os.remove(tmp_curr) # Immediate deletion to save disk space.

                # DUAL-LOOKAHEAD: Compare current vs next segment to find the cut point.
                if curr_num < total_count:
                    next_seg = segments[i+1]
                    fd2, tmp_next = tempfile.mkstemp(suffix=".wav")
                    os.close(fd2)
                    # Transcribe next segment to establish the 'Right Anchor'.
                    subprocess.run(["ffmpeg", "-y", "-ss", str(next_seg['start']), "-t", str(next_seg['dur']), "-i", current_audio_workfile, "-ar", "16000", "-ac", "1", tmp_next], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    transcript_next = ai_model.transcribe(tmp_next)["text"].strip()
                    os.remove(tmp_next) 
                    
                    # PHASE A: Calculate initial split via sliding window ratio analysis.
                    split_idx, match_score = get_refined_split_pos(transcript_curr, transcript_next, remaining_script, 45, 400, 5.5, seg['dur'])
                    
                    # PHASE B: PHONETIC TAIL REFINEMENT.
                    # Forces the cut point to expand until it captures the exact words
                    # heard at the end of the current audio segment.
                    initial_slice = remaining_script[:split_idx].strip()
                    t_curr_end = " ".join(transcript_curr.split()[-2:]) 
                    
                    if match_score >= SCORE_THRESHOLD and t_curr_end:
                        if t_curr_end not in initial_slice:
                            # Search forward to find missing trailing words in the master script.
                            search_limit = 400 if seg['dur'] >= 5.5 else 45
                            found_pos = remaining_script[:search_limit].rfind(t_curr_end)
                            if found_pos != -1:
                                new_end_pos = found_pos + len(t_curr_end)
                                if new_end_pos > split_idx:
                                    split_idx = new_end_pos 
                else:
                    # EOF handling: Assign all remaining script text to the final segment.
                    split_idx = len(remaining_script)
                    match_score = 1.0

                # --- SYNCHRONIZATION ROBUSTNESS LOGIC ---
                # If alignment confidence is low, we freeze the script pointer.
                # This ensures an AI hallucination doesn't misalign future valid audio.
                if match_score >= SCORE_THRESHOLD:
                    # Successful match: Extract normalized slice from master.
                    final_segment_text = remaining_script[:split_idx].strip()
                    actual_consumed = split_idx
                else:
                    # Failed match: Write empty file as a placeholder for human review.
                    final_segment_text = ""
                    actual_consumed = 0
                
                # PERSISTENCE: newline='' ensures binary consistency for future find() sync.
                with open(txt_file_path, "w", encoding="utf-8", newline='') as f:
                    f.write(final_segment_text)
                
                # Advance script pointer based on validated consumption (0 for failures).
                remaining_script = remaining_script[actual_consumed:].strip()

            # --- PROGRESS MONITORING ---
            # Console visualization with simplified text (newlines removed for logging).
            preview = final_segment_text.replace("\n", " ")
            preview = (preview[:37] + "...") if len(preview) > 40 else preview
            log_header = f"[{curr_num:04d}/{total_count:04d}] Score:{match_score:.2f} -> "
            
            if final_segment_text:
                print(f"{log_header}{preview}")
            else:
                # Direct feedback when synchronization is intentionally suspended.
                print(f"{log_header}HOLD (Sync confidence too low - pointer maintained)")

            # Generate SRT block formatting for final assembly.
            start_time_srt = format_srt_time(seg['start'])
            end_time_srt = format_srt_time(seg['end'])
            srt_data.append(f"{curr_num}\n{start_time_srt} --> {end_time_srt}\n{final_segment_text}\n")

        # --- STEP 5: FINAL SRT ASSEMBLY ---
        # Writes all validated blocks into a single specification-compliant subtitle file.
        with open(output_srt, "w", encoding="utf-8") as f:
            f.write("\n".join(srt_data))
        print(f"\nSUCCESS:\n- Saved Fragments: {output_dir}/\n- Master SRT File: {output_srt}")

    finally:
        # --- STEP 6: RESOURCE CLEANUP ---
        # Ensures heavy temporary WAV workfiles are deleted regardless of exit status.
        if temp_wav_file and os.path.exists(temp_wav_file):
            os.remove(temp_wav_file)
            print("CLEANUP: Normalization workfile purged.")

if __name__ == "__main__":
    # Provides support for clean termination via Ctrl+C.
    try:
        run()
    except KeyboardInterrupt:
        print("\nProcess aborted by user command.")
        sys.exit(0)

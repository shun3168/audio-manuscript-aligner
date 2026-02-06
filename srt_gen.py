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
# CONFIGURATION: ALIGNMENT & SYNCHRONIZATION THRESHOLDS
# ==============================================================================

# This threshold determines the minimum confidence for script-audio alignment.
# If the initial similarity is below 0.5, script consumption is halted 
# to prevent the "avalanche effect" (cascading synchronization errors).
SCORE_THRESHOLD = 0.5

# ==============================================================================
# UTILITY: SRT TIMESTAMP GENERATOR
# ==============================================================================

def format_srt_time(seconds):
    """
    PURPOSE: Converts a raw float of seconds into the SRT standard time string.
    LOGIC: Uses timedelta to extract units and formats milliseconds with a comma.
    FORMAT: HH:MM:SS,mmm (e.g., 00:00:05,123)
    """
    td = timedelta(seconds=seconds)
    total_sec = int(td.total_seconds())
    msec = int(td.microseconds / 1000)
    return f"{total_sec//3600:02}:{(total_sec%3600)//60:02}:{total_sec%60:02},{msec:03}"

# ==============================================================================
# CORE ALGORITHM: SLIDING WINDOW PHONETIC ALIGNMENT
# ==============================================================================

def get_refined_split_pos(curr_heard, next_heard, script_segment, base_limit, extended_limit, long_wav_sec, dur):
    """
    PURPOSE: Calculates the optimal character index in the script for splitting.
    
    LOGIC:
    1. ANCHORING: Extracts the last 2 words of the current clip and first 2 of the next.
    2. SLIDING SEARCH: Moves through the script slice to find the point 'i' where 
       the text before 'i' matches the current clip's end, and text after 'i' 
       matches the next clip's start.
    3. SCORING: Returns the index with the highest average similarity ratio.
    """
    # Extract phonetic anchors (markers) for the boundary
    words_next = next_heard.split()[:2]
    words_curr = curr_heard.split()[-2:]
    t_next = " ".join(words_next)
    t_curr = " ".join(words_curr)
    
    # Range is dynamically expanded for longer audio segments to maintain sync.
    limit = extended_limit if dur >= long_wav_sec else base_limit
    segment = script_segment[:limit]
    
    best_pos, max_score = 0, -1
    # Iterate through the segment to find the best phonetic handshake
    for i in range(len(segment) + 1):
        # s_left verifies the trailing edge / s_right verifies the leading edge
        s_left = SequenceMatcher(None, t_curr, segment[max(0, i-len(t_curr)):i].strip()).ratio() if t_curr else 0
        s_right = SequenceMatcher(None, t_next, segment[i:i+len(t_next)].strip()).ratio() if t_next else 0
        
        # Average the two scores for the final confidence ranking
        score = (s_left + s_right) / 2
        if score > max_score:
            max_score, best_pos = score, i
            
    return best_pos, max_score

# ==============================================================================
# MAIN PROCESSING ENGINE
# ==============================================================================

def run():
    # --- STEP 1: PATH RESOLUTION & ARGUMENT VALIDATION ---
    # Check if the mandatory audio and script arguments are provided via CLI.
    if len(sys.argv) < 3:
        print("USAGE: python srt_gen.py <audio_file> <script_file>")
        sys.exit(1)

    audio_in = sys.argv[1]
    script_arg = sys.argv[2]

    # Stop execution if the source audio file is missing.
    if not os.path.exists(audio_in):
        print(f"ERROR: Audio file '{audio_in}' not found.")
        sys.exit(1)

    # Resolution logic for script file extensions (.txt).
    script_in = script_arg
    if not os.path.exists(script_in):
        if os.path.exists(script_arg + ".txt"):
            script_in = script_arg + ".txt"
        else:
            print(f"ERROR: Script '{script_arg}' not found.")
            sys.exit(1)

    # Define output naming conventions based on input filenames.
    audio_name = os.path.splitext(os.path.basename(audio_in))[0]
    script_name = os.path.splitext(os.path.basename(script_in))[0]
    output_dir = script_name
    output_srt = audio_name + ".srt"

    # --- STEP 2: AUDIO SIGNAL STANDARDIZATION ---
    # Normalizes input to 16kHz Mono WAV. This ensures consistent results
    # in both FFmpeg silence detection and OpenAI Whisper transcription.
    current_audio_workfile = audio_in
    temp_wav_file = None
    if not audio_in.lower().endswith(".wav"):
        print(f"NOTICE: Creating standardized 16kHz Mono WAV workfile...")
        fd, temp_wav_file = tempfile.mkstemp(suffix=".wav")
        os.close(fd) 
        # ac 1 = Mono, ar 16000 = 16kHz sample rate.
        conv_cmd = ["ffmpeg", "-y", "-i", audio_in, "-ac", "1", "-ar", "16000", temp_wav_file]
        subprocess.run(conv_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        current_audio_workfile = temp_wav_file

    try:
        # Create directory for caching text fragments (persistence).
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # SCRIPT NORMALIZATION: Collapses all newlines and multiple spaces into 
        # a single whitespace-separated string for stable character indexing.
        with open(script_in, "r", encoding="utf-8") as f:
            master_script = " ".join(f.read().split())

        # --- STEP 3: VAD (VOICE ACTIVITY DETECTION) ---
        print(f"STEP 1: Mapping speech intervals via FFmpeg silencedetect...")
        # Silence threshold: -30dB, minimum duration: 0.5s.
        log_cmd = ["ffmpeg", "-i", current_audio_workfile, "-af", "silencedetect=noise=-30dB:d=0.5", "-f", "null", "-"]
        # errors='replace' avoids crashes on locale-specific log encoding.
        result = subprocess.run(log_cmd, stderr=subprocess.PIPE, text=True, errors="replace")
        
        # Regex patterns to isolate start/end timestamps from the logs.
        silence_starts = re.findall(r"silence_start: ([\d\.]+)", result.stderr)
        silence_ends = re.findall(r"silence_end: ([\d\.]+)", result.stderr)

        # Calculate durations and build the segment metadata list.
        segments = []
        for i in range(len(silence_ends)):
            start = float(silence_ends[i])
            # Handle end of the file for the final segment.
            end = float(silence_starts[i+1]) if i + 1 < len(silence_starts) else start + 5.0
            if end - start > 0.1: # Discard micro-noise artifacts.
                segments.append({"start": start, "end": end, "dur": end - start})

        total_count = len(segments)
        print(f"Total speech segments detected: {total_count}")
        
        ai_model = None
        remaining_script = master_script
        srt_data = []

        # --- STEP 4: TRANSCRIPTION & ALIGNMENT LOOP ---
        for i, seg in enumerate(segments):
            curr_num = i + 1
            # Primary key for fragment files based on start-time (milliseconds).
            timestamp_ms = int(seg['start'] * 1000)
            txt_filename = f"{timestamp_ms:09d}.txt"
            txt_file_path = os.path.join(output_dir, txt_filename)
            
            final_segment_text = ""
            match_score = 0.0 

            # STATE RECOVERY: Attempt to resume from existing fragments.
            if os.path.exists(txt_file_path):
                with open(txt_file_path, "r", encoding="utf-8") as f:
                    final_segment_text = f.read()
                
                # If a valid match exists, advance the master script pointer.
                if final_segment_text:
                    find_idx = remaining_script.find(final_segment_text)
                    if find_idx != -1: 
                        remaining_script = remaining_script[find_idx + len(final_segment_text):].strip()
            else:
                # LAZY LOADING: Initialize the heavy AI model only when needed.
                if ai_model is None:
                    print("STEP 2: Initializing Whisper AI (Turbo model)...")
                    ai_model = whisper.load_model("turbo")

                # EXTRACTION: Write isolated audio chunk for Whisper processing.
                fd1, tmp_curr = tempfile.mkstemp(suffix=".wav")
                os.close(fd1)
                subprocess.run(["ffmpeg", "-y", "-ss", str(seg['start']), "-t", str(seg['dur']), "-i", current_audio_workfile, "-ar", "16000", "-ac", "1", tmp_curr], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # Transcribe current audio chunk to text.
                transcript_curr = ai_model.transcribe(tmp_curr)["text"].strip()
                os.remove(tmp_curr) 

                # Determine the split boundary by looking at the next audio segment.
                if curr_num < total_count:
                    next_seg = segments[i+1]
                    fd2, tmp_next = tempfile.mkstemp(suffix=".wav")
                    os.close(fd2)
                    # Transcribe next chunk to establish the Right Anchor.
                    subprocess.run(["ffmpeg", "-y", "-ss", str(next_seg['start']), "-t", str(next_seg['dur']), "-i", current_audio_workfile, "-ar", "16000", "-ac", "1", tmp_next], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    transcript_next = ai_model.transcribe(tmp_next)["text"].strip()
                    os.remove(tmp_next) 
                    
                    # 1. INITIAL ALIGNMENT (origtxt Logic): Determine split point and initial score.
                    split_idx, match_score = get_refined_split_pos(transcript_curr, transcript_next, remaining_script, 45, 400, 5.5, seg['dur'])
                else:
                    # EOF handler: Absorb all remaining text.
                    split_idx = len(remaining_script)
                    match_score = 1.0

                # 2. STRICT SYNCHRONIZATION DECISION (origtxt Logic):
                # We only proceed to consume the script if the initial match score passes the threshold.
                if match_score >= SCORE_THRESHOLD:
                    # Initial slice from the script based on phonetic alignment.
                    final_segment_text = remaining_script[:split_idx].strip()
                    
                    # 3. TAIL-ANCHOR CORRECTION (origtxt Logic):
                    # Validates if the Whisper end-anchor is present in the current slice.
                    # If not, it extends the slice to include the actual words heard.
                    t_curr_end = " ".join(transcript_curr.split()[-2:])
                    if t_curr_end and t_curr_end not in final_segment_text:
                        search_limit = 400 if seg['dur'] >= 5.5 else 45
                        found_pos = remaining_script[:search_limit].rfind(t_curr_end)
                        if found_pos != -1:
                            new_end_pos = found_pos + len(t_curr_end)
                            # Only update if the correction extends the slice forward.
                            if new_end_pos > split_idx:
                                final_segment_text = remaining_script[:new_end_pos].strip()
                                split_idx = new_end_pos
                    
                    # Set the amount of script to be consumed globally.
                    actual_consumed = split_idx
                else:
                    # SYNC FAILURE: Leave text empty and freeze the script pointer.
                    final_segment_text = ""
                    actual_consumed = 0
                
                # PERSISTENCE: newline='' ensures binary consistency for find() synchronization.
                with open(txt_file_path, "w", encoding="utf-8", newline='') as f:
                    f.write(final_segment_text)
                
                # Advance the global master script pointer.
                remaining_script = remaining_script[actual_consumed:].strip()

            # --- MONITORING: CONSOLE FEEDBACK ---
            # Remove newlines for the terminal preview log.
            preview = final_segment_text.replace("\n", " ")
            preview = (preview[:37] + "...") if len(preview) > 40 else preview
            log_msg = f"[{curr_num:04d}/{total_count:04d}] Score:{match_score:.2f} -> "
            # Visual indicator of successful match vs sync hold.
            print(f"{log_msg}{preview}" if final_segment_text else f"{log_msg}SKIP (Pointer Held)")

            # Format the entry into the final SRT data buffer.
            start_time_srt = format_srt_time(seg['start'])
            end_time_srt = format_srt_time(seg['end'])
            srt_data.append(f"{curr_num}\n{start_time_srt} --> {end_time_srt}\n{final_segment_text}\n")

        # --- STEP 5: FINAL SRT ASSEMBLY ---
        # Writes the combined buffer to the final .srt file.
        with open(output_srt, "w", encoding="utf-8") as f:
            f.write("\n".join(srt_data))
        print(f"\nFINISH:\n- Fragments saved in: {output_dir}/\n- Master Subtitles: {output_srt}")

    finally:
        # --- STEP 6: CLEANUP ---
        # Ensure heavy temporary audio workfiles are purged from disk.
        if temp_wav_file and os.path.exists(temp_wav_file):
            os.remove(temp_wav_file)
            print("CLEANUP: Process workfile removed.")

if __name__ == "__main__":
    # Clean exit handling for KeyboardInterrupt (Ctrl+C).
    try:
        run()
    except KeyboardInterrupt:
        print("\nProcess aborted by user.")
        sys.exit(0)

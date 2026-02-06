import os
import subprocess
import re
import whisper
import io
import sys
import tempfile
from difflib import SequenceMatcher
from datetime import timedelta

# Threshold for the similarity score (0.0 to 1.0).
# Alignment is considered reliable if the score is above this value.
SCORE_THRESHOLD = 0.5

def format_srt_time(seconds):
    """
    Converts raw seconds (float) into the standard SRT subtitle timestamp format.
    Format: HH:MM:SS,mmm (e.g., 00:01:02,500)
    """
    td = timedelta(seconds=seconds)
    total_sec = int(td.total_seconds())
    msec = int(td.microseconds / 1000)
    return f"{total_sec//3600:02}:{(total_sec%3600)//60:02}:{total_sec%60:02},{msec:03}"

def get_refined_split_pos(curr_heard, next_heard, script_segment, base_limit, extended_limit, long_wav_sec, dur):
    """
    Core Alignment Algorithm: Sliding Window Anchor Matching.
    
    This function finds the best 'cut point' in the master script by comparing
    Whisper's transcribed text with the original script text.
    
    1. It identifies 'anchors': the end of the current clip and the start of the next.
    2. It slides through the script and calculates a similarity ratio at every possible character index.
    3. The position with the highest average similarity for both anchors is returned.
    """
    # Use the last 2 words of current clip and first 2 words of next clip as anchors
    words_next = next_heard.split()[:2]
    words_curr = curr_heard.split()[-2:]
    t_next = " ".join(words_next)
    t_curr = " ".join(words_curr)
    
    # Adjust search range based on audio duration to prevent out-of-sync drifting
    limit = extended_limit if dur >= long_wav_sec else base_limit
    segment = script_segment[:limit]
    
    best_pos, max_score = 0, -1
    
    # Iterate through every character in the search segment to find the optimal split
    for i in range(len(segment) + 1):
        # Calculate similarity for the 'left' side (current segment end)
        s_left = SequenceMatcher(None, t_curr, segment[max(0, i-len(t_curr)):i].strip()).ratio() if t_curr else 0
        # Calculate similarity for the 'right' side (next segment start)
        s_right = SequenceMatcher(None, t_next, segment[i:i+len(t_next)].strip()).ratio() if t_next else 0
        
        # Combined score: 1.0 means perfect alignment with both anchors
        score = (s_left + s_right) / 2
        if score > max_score:
            max_score, best_pos = score, i
            
    return best_pos, max_score

def run():
    # --- Phase 1: Environment & Argument Validation ---
    if len(sys.argv) < 3:
        print("USAGE: python srt_gen.py <audio_file> <script_file>")
        sys.exit(1)

    audio_in = sys.argv[1]
    script_arg = sys.argv[2]

    # Check if the source audio exists
    if not os.path.exists(audio_in):
        print(f"ERROR: Audio file '{audio_in}' not found.")
        sys.exit(1)

    # Resolve script path (allows user to omit '.txt' in command line)
    script_in = script_arg
    if not os.path.exists(script_in):
        if os.path.exists(script_arg + ".txt"):
            script_in = script_arg + ".txt"
        else:
            print(f"ERROR: Script '{script_arg}' not found.")
            sys.exit(1)

    # Initialize naming conventions for output files and directories
    audio_name = os.path.splitext(os.path.basename(audio_in))[0]
    script_name = os.path.splitext(os.path.basename(script_in))[0]
    output_dir = script_name
    output_srt = audio_name + ".srt"

    # --- Phase 2: Audio Normalization (FFmpeg) ---
    # Whisper works most accurately with 16kHz mono WAV files.
    # We create a temporary high-compatibility file if the input is in another format.
    current_audio_workfile = audio_in
    temp_wav_file = None
    if not audio_in.lower().endswith(".wav"):
        print(f"NOTICE: Converting input to 16kHz mono WAV for optimal AI processing...")
        fd, temp_wav_file = tempfile.mkstemp(suffix=".wav")
        os.close(fd) # Close file descriptor; subprocess will handle the file path
        conv_cmd = ["ffmpeg", "-y", "-i", audio_in, "-ac", "1", "-ar", "16000", temp_wav_file]
        subprocess.run(conv_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        current_audio_workfile = temp_wav_file

    try:
        # Create storage for individual text fragments (useful for manual correction)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Read the master script and normalize all whitespace/newlines into single spaces
        with open(script_in, "r", encoding="utf-8") as f:
            master_script = " ".join(f.read().split())

        # --- Phase 3: Speech Segmentation via Silence Detection ---
        print(f"STEP 1: Analyzing audio for speech intervals...")
        # We detect silence to find gaps between sentences. -30dB is the noise floor.
        log_cmd = ["ffmpeg", "-i", current_audio_workfile, "-af", "silencedetect=noise=-30dB:d=0.5", "-f", "null", "-"]
        
        # CROSS-OS COMPATIBILITY: We use errors="replace" for the stderr stream.
        # This prevents crashes if FFmpeg outputs non-UTF8 characters in Windows environments.
        result = subprocess.run(log_cmd, stderr=subprocess.PIPE, text=True, errors="replace")
        
        # Regex to extract timestamps where silence starts and ends
        silence_starts = re.findall(r"silence_start: ([\d\.]+)", result.stderr)
        silence_ends = re.findall(r"silence_end: ([\d\.]+)", result.stderr)

        # Build a list of active speech segments (the parts between silences)
        segments = []
        for i in range(len(silence_ends)):
            start = float(silence_ends[i])
            # Set end point to next silence start, or +5s if it's the final segment
            end = float(silence_starts[i+1]) if i + 1 < len(silence_starts) else start + 5.0
            if end - start > 0.1: # Ignore micro-segments or noise
                segments.append({"start": start, "end": end, "dur": end - start})

        print(f"Total speech segments detected: {len(segments)}")
        
        ai_model = None
        remaining_script = master_script
        srt_data = []

        # --- Phase 4: Main Transcription & Alignment Loop ---
        for i, seg in enumerate(segments):
            timestamp_ms = int(seg['start'] * 1000)
            txt_file_path = os.path.join(output_dir, f"{timestamp_ms:09d}.txt")
            
            final_segment_text = ""
            # RESUME FEATURE: If a text fragment already exists, skip AI processing to save time
            if os.path.exists(txt_file_path):
                with open(txt_file_path, "r", encoding="utf-8") as f:
                    final_segment_text = f.read().strip()
                print(f"[{i+1:03d}] SKIP: {os.path.basename(txt_file_path)} (Found cached fragment)")
                # Move script pointer forward
                find_idx = remaining_script.find(final_segment_text)
                if find_idx != -1: 
                    remaining_script = remaining_script[find_idx + len(final_segment_text):].strip()
            else:
                # Load Whisper AI model on-demand (lazy loading)
                if ai_model is None:
                    print("STEP 2: Loading OpenAI Whisper model (Turbo)...")
                    ai_model = whisper.load_model("turbo")

                # STABILITY FIX: Extract segment to a physical temp file.
                # Whisper's transcribe() often fails with raw ByteIO streams on certain OSs.
                fd1, tmp_curr = tempfile.mkstemp(suffix=".wav")
                os.close(fd1)
                subprocess.run(["ffmpeg", "-y", "-ss", str(seg['start']), "-t", str(seg['dur']), "-i", current_audio_workfile, "-ar", "16000", "-ac", "1", tmp_curr], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # Perform AI speech-to-text on the current chunk
                transcript_curr = ai_model.transcribe(tmp_curr)["text"].strip()
                os.remove(tmp_curr) # Cleanup audio chunk immediately to free disk space

                # ALIGNMENT LOGIC: Compare current segment with the next one to find the boundary
                if i + 1 < len(segments):
                    next_seg = segments[i+1]
                    fd2, tmp_next = tempfile.mkstemp(suffix=".wav")
                    os.close(fd2)
                    subprocess.run(["ffmpeg", "-y", "-ss", str(next_seg['start']), "-t", str(next_seg['dur']), "-i", current_audio_workfile, "-ar", "16000", "-ac", "1", tmp_next], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    transcript_next = ai_model.transcribe(tmp_next)["text"].strip()
                    os.remove(tmp_next) 
                    
                    # Phase A: Get base split position using the anchor similarity algorithm
                    split_idx, match_score = get_refined_split_pos(transcript_curr, transcript_next, remaining_script, 45, 400, 5.5, seg['dur'])
                    
                    # Phase B (Ported from origtxt.py): Tail-Anchor Correction.
                    # This independently verifies if the Whisper-detected ending words are in the text.
                    # If they are missing due to a low similarity score, we force an extension.
                    initial_text = remaining_script[:split_idx].strip()
                    t_curr_end = " ".join(transcript_curr.split()[-2:]) # Last 2 words from Whisper
                    
                    if match_score >= SCORE_THRESHOLD and t_curr_end:
                        # If the script cut doesn't contain the words Whisper actually heard:
                        if t_curr_end not in initial_text:
                            # Search for these words slightly deeper in the script
                            search_limit = 400 if seg['dur'] >= 5.5 else 45
                            found_pos = remaining_script[:search_limit].rfind(t_curr_end)
                            if found_pos != -1:
                                new_end_pos = found_pos + len(t_curr_end)
                                # Only adjust if it actually increases the text length
                                if new_end_pos > split_idx:
                                    split_idx = new_end_pos 
                else:
                    # Final segment: absorb all remaining script text
                    split_idx = len(remaining_script)
                    match_score = 1.0

                # Finalize the fragment and save to disk
                final_segment_text = remaining_script[:split_idx].strip()
                with open(txt_file_path, "w", encoding="utf-8") as f:
                    f.write(final_segment_text)
                
                # Consume the master script by moving the pointer
                remaining_script = remaining_script[split_idx:].strip()

                # Visual Progress Logging for the user
                preview = final_segment_text.replace("\n", " ")
                preview = (preview[:37] + "...") if len(preview) > 40 else preview
                print(f"[{i+1:03d}] SAVED: {os.path.basename(txt_file_path)} -> \"{preview}\" (Score: {match_score:.2f})")

            # Append the result to the SRT data list
            start_time_srt = format_srt_time(seg['start'])
            end_time_srt = format_srt_time(seg['end'])
            srt_data.append(f"{i+1}\n{start_time_srt} --> {end_time_srt}\n{final_segment_text}\n")

        # --- Phase 5: Final Export & Cleanup ---
        with open(output_srt, "w", encoding="utf-8") as f:
            f.write("\n".join(srt_data))
        print(f"\nFINISH:\n- Script fragments saved in: {output_dir}/\n- Final Subtitle file created: {output_srt}")

    finally:
        # Crucial Cleanup: Delete the master temporary WAV file even if an error occurs
        if temp_wav_file and os.path.exists(temp_wav_file):
            os.remove(temp_wav_file)
            print("CLEANUP: Process workfile deleted.")

if __name__ == "__main__":
    # Execute with keyboard interrupt protection
    try:
        run()
    except KeyboardInterrupt:
        print("\nAborted by user.")
        sys.exit(0)

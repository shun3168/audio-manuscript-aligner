import os
import subprocess
import re
import whisper
import io
import sys
import tempfile
import wave
from difflib import SequenceMatcher
from datetime import timedelta

# --- Configuration Constants ---
# BASE_LIMIT: Default search range for short audio clips.
BASE_LIMIT = 45 
# EXTENDED_LIMIT: Expanded search range to prevent text "suffocation" in long or complex clips.
EXTENDED_LIMIT = 400 
# LONG_WAV_SEC: Threshold (seconds) to switch from BASE to EXTENDED limit.
LONG_WAV_SEC = 5.5 
# SCORE_THRESHOLD: Minimum confidence score (0.0-1.0). Below this, the AI might be hallucinating.
SCORE_THRESHOLD = 0.5 
# -------------------------------

def format_srt_time(seconds):
    """
    Converts raw seconds into the standard SRT time format: HH:MM:SS,mmm.
    Example: 61.503 -> 00:01:01,503
    """
    td = timedelta(seconds=seconds)
    total_sec = int(td.total_seconds())
    msec = int(td.microseconds / 1000)
    return f"{total_sec//3600:02}:{(total_sec%3600)//60:02}:{total_sec%60:02},{msec:03}"

def get_refined_split_pos(curr_heard, next_heard, script_segment, base_limit, extended_limit, long_wav_sec, dur):
    """
    Core Alignment Logic:
    Determines where the current audio clip ends within the script by comparing 
    AI transcripts (current and next) against the master script.
    """
    # Use the first 2 words of the next segment and last 2 of the current as "anchors".
    words_next = next_heard.split()[:2]
    words_curr = curr_heard.split()[-2:]
    t_next = " ".join(words_next)
    t_curr = " ".join(words_curr)
    
    # Decide search range based on clip duration.
    limit = extended_limit if dur >= long_wav_sec else base_limit
    segment = script_segment[:limit]
    
    best_pos, max_score = 0, -1
    # Iterate through the script segment to find the point that best splits 
    # the 'current heard' text from the 'next heard' text.
    for i in range(len(segment) + 1):
        # Match probability for the text preceding the split point.
        s_left = SequenceMatcher(None, t_curr, segment[max(0, i-len(t_curr)):i].strip()).ratio() if t_curr else 0
        # Match probability for the text following the split point.
        s_right = SequenceMatcher(None, t_next, segment[i:i+len(t_next)].strip()).ratio() if t_next else 0
        
        # Average score determines the best fit for the boundary.
        score = (s_left + s_right) / 2
        if score > max_score:
            max_score, best_pos = score, i
            
    return best_pos, max_score

def run():
    # --- 1. Argument and Environment Validation ---
    if len(sys.argv) < 3:
        print("USAGE: python srt_gen.py <audio_file> <script_file>")
        sys.exit(1)

    audio_in = sys.argv[1]
    script_arg = sys.argv[2]

    if not os.path.exists(audio_in):
        print(f"ERROR: Audio file '{audio_in}' not found.")
        sys.exit(1)

    script_in = script_arg
    if not os.path.exists(script_in):
        if os.path.exists(script_arg + ".txt"):
            script_in = script_arg + ".txt"
        else:
            print(f"ERROR: Script '{script_arg}' not found.")
            sys.exit(1)

    # Prepare naming conventions for folders and SRT files.
    audio_name = os.path.splitext(os.path.basename(audio_in))[0]
    script_name = os.path.splitext(os.path.basename(script_in))[0]
    output_dir = script_name
    output_srt = audio_name + ".srt"

    # --- 2. Temporary WAV Conversion ---
    # Convert non-WAV files to 16kHz mono WAV for optimal Whisper and silence detection performance.
    current_audio_workfile = audio_in
    temp_wav_file = None
    if not audio_in.lower().endswith(".wav"):
        print(f"NOTICE: Converting {audio_in} to a temporary WAV for precise processing...")
        temp_wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        conv_cmd = ["ffmpeg", "-y", "-i", audio_in, "-ac", "1", "-ar", "16000", temp_wav_file]
        subprocess.run(conv_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        current_audio_workfile = temp_wav_file

    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Load the master script and normalize whitespace.
        with open(script_in, "r", encoding="utf-8") as f:
            master_script = " ".join(f.read().split())

        # --- 3. Speech Interval Detection ---
        # Use FFmpeg's silencedetect filter to find non-silent segments.
        print(f"STEP 1: Scanning {audio_in} for speech intervals...")
        log_cmd = ["ffmpeg", "-i", current_audio_workfile, "-af", "silencedetect=noise=-30dB:d=0.5", "-f", "null", "-"]
        result = subprocess.run(log_cmd, stderr=subprocess.PIPE, text=True, encoding="utf-8")
        
        silence_starts = re.findall(r"silence_start: ([\d\.]+)", result.stderr)
        silence_ends = re.findall(r"silence_end: ([\d\.]+)", result.stderr)

        # Build segments list: (start, end, duration).
        segments = []
        for i in range(len(silence_ends)):
            start = float(silence_ends[i])
            # If there's no next silence, assume a 5.0s tail.
            end = float(silence_starts[i+1]) if i + 1 < len(silence_starts) else start + 5.0
            if end - start > 0.1:
                segments.append({"start": start, "end": end, "dur": end - start})

        print(f"Total segments found: {len(segments)}")
        
        ai_model = None
        remaining_script = master_script
        srt_data = []

        # --- 4. Main Alignment Loop ---
        for i, seg in enumerate(segments):
            timestamp_ms = int(seg['start'] * 1000)
            txt_file_path = os.path.join(output_dir, f"{timestamp_ms:09d}.txt")
            
            # RESUME FEATURE: If the txt file already exists, use it and move forward.
            if os.path.exists(txt_file_path):
                with open(txt_file_path, "r", encoding="utf-8") as f:
                    final_segment_text = f.read().strip()
                print(f"[{i+1:03d}] SKIP: {os.path.basename(txt_file_path)} (Cached)")
                find_idx = remaining_script.find(final_segment_text)
                if find_idx != -1: 
                    remaining_script = remaining_script[find_idx + len(final_segment_text):].strip()
            else:
                # Lazy load the Whisper model only when needed.
                if ai_model is None:
                    print("STEP 2: Loading Whisper AI...")
                    ai_model = whisper.load_model("turbo")

                # Extract audio slice via pipe to avoid disk I/O overhead.
                extract_cmd = ["ffmpeg", "-ss", str(seg['start']), "-t", str(seg['dur']), "-i", current_audio_workfile, "-f", "wav", "-ar", "16000", "-ac", "1", "pipe:1"]
                audio_buffer = subprocess.run(extract_cmd, capture_output=True).stdout
                transcript_curr = ai_model.transcribe(io.BytesIO(audio_buffer))["text"].strip()

                current_limit = EXTENDED_LIMIT if seg['dur'] >= LONG_WAV_SEC else BASE_LIMIT

                # Peek at the next segment to determine the exact boundary.
                if i + 1 < len(segments):
                    next_seg = segments[i+1]
                    next_cmd = ["ffmpeg", "-ss", str(next_seg['start']), "-t", str(next_seg['dur']), "-i", current_audio_workfile, "-f", "wav", "-ar", "16000", "-ac", "1", "pipe:1"]
                    next_buffer = subprocess.run(next_cmd, capture_output=True).stdout
                    transcript_next = ai_model.transcribe(io.BytesIO(next_buffer))["text"].strip()
                    
                    split_idx, match_score = get_refined_split_pos(transcript_curr, transcript_next, remaining_script, BASE_LIMIT, EXTENDED_LIMIT, LONG_WAV_SEC, seg['dur'])

                    # RECOVERY LOGIC: If confidence is low, re-scan with a wider window (EXTENDED_LIMIT).
                    if match_score < SCORE_THRESHOLD and current_limit == BASE_LIMIT:
                        current_limit = EXTENDED_LIMIT
                        split_idx, match_score = get_refined_split_pos(transcript_curr, transcript_next, remaining_script, BASE_LIMIT, EXTENDED_LIMIT, LONG_WAV_SEC, seg['dur'])
                else:
                    # Final clip: consume all remaining text.
                    split_idx = len(remaining_script)
                    match_score = 1.0

                # --- Independent Pointer Verification ---
                # Check if the calculated 'split_idx' actually covers everything AI heard.
                if match_score < SCORE_THRESHOLD and i + 1 < len(segments):
                    # NOISE PROTECTION: If score is still too low, treat as non-speech. 
                    # Set text to empty and hold the script pointer (split_idx = 0).
                    final_segment_text = ""
                    split_idx = 0
                else:
                    # POINTER SYNC: Ensure skipping Whisper's summarization habits.
                    final_segment_text = remaining_script[:split_idx].strip()
                    t_curr_end = " ".join(transcript_curr.split()[-2:])
                    
                    # If AI heard words that are beyond our current split_idx, 
                    # we force the pointer forward to capture that missed text into the current file.
                    if t_curr_end and (t_curr_end not in final_segment_text):
                        found_pos = remaining_script[:current_limit].rfind(t_curr_end)
                        if found_pos != -1:
                            new_split = found_pos + len(t_curr_end)
                            if new_split > split_idx:
                                final_segment_text = remaining_script[:new_split].strip()
                                split_idx = new_split

                # Write the finalized text to disk.
                with open(txt_file_path, "w", encoding="utf-8") as f:
                    f.write(final_segment_text)
                
                # Advance the master script pointer.
                remaining_script = remaining_script[split_idx:].strip()
                status = "SAVED" if final_segment_text else "SKIPPED"
                print(f"[{i+1:03d}] {status} (Score: {match_score:.2f}): {os.path.basename(txt_file_path)} -> {final_segment_text[:40]}...")

            # --- 5. SRT Data Compilation ---
            # Append non-empty segments to the SRT output list.
            if final_segment_text:
                start_time_srt = format_srt_time(seg['start'])
                end_time_srt = format_srt_time(seg['end'])
                # Use len(srt_data)+1 to ensure continuous numbering even if clips were skipped.
                srt_data.append(f"{len(srt_data)+1}\n{start_time_srt} --> {end_time_srt}\n{final_segment_text}\n")

        # --- 6. Final Export ---
        with open(output_srt, "w", encoding="utf-8") as f:
            f.write("\n".join(srt_data))
        print(f"\nFINISH:\n- Fragments: {output_dir}/\n- Subtitles: {output_srt}")

    finally:
        # Cleanup temporary files regardless of success or failure.
        if temp_wav_file and os.path.exists(temp_wav_file):
            os.remove(temp_wav_file)
            print("CLEANUP: Temporary WAV deleted.")

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\nOperation aborted by user.")
        sys.exit(0)

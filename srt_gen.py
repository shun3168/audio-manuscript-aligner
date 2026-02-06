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
    Converts a float duration (seconds) into the standard SRT timestamp format.
    Required format: HH:MM:SS,mmm (milliseconds separated by a comma).
    """
    td = timedelta(seconds=seconds)
    total_sec = int(td.total_seconds())
    msec = int(td.microseconds / 1000)
    return f"{total_sec//3600:02}:{(total_sec%3600)//60:02}:{total_sec%60:02},{msec:03}"

def get_refined_split_pos(curr_heard, next_heard, script_segment, base_limit, extended_limit, long_wav_sec, dur):
    """
    ALGORITHM: Text-Audio Alignment via Anchor Matching.
    
    This function determines the optimal split point in the master script to synchronize
    it with the detected audio segments.
    
    1. Anchors: It takes the last 2 words of the current transcription and the first 2 
       words of the next transcription as 'anchors'.
    2. Scoring: It iterates through the master script and calculates a similarity score 
       (0.0 to 1.0) based on how well these anchors match the text at each position.
    3. Resilience: Using SequenceMatcher.ratio() allows it to handle minor 
       transcription errors or script variations without failing.
    """
    # Define anchor phrases from the current and subsequent transcriptions
    words_next = next_heard.split()[:2]
    words_curr = curr_heard.split()[-2:]
    t_next = " ".join(words_next)
    t_curr = " ".join(words_curr)
    
    # Range Limiter: Prevent the search from drifting too far into the script.
    # Longer audio segments (dur >= long_wav_sec) allow for a larger search buffer.
    limit = extended_limit if dur >= long_wav_sec else base_limit
    segment = script_segment[:limit]
    
    best_pos, max_score = 0, -1
    
    # Sliding window search to find the position with the highest cumulative similarity
    for i in range(len(segment) + 1):
        # Match current end-anchor with script slice ending at 'i'
        s_left = SequenceMatcher(None, t_curr, segment[max(0, i-len(t_curr)):i].strip()).ratio() if t_curr else 0
        # Match next start-anchor with script slice starting at 'i'
        s_right = SequenceMatcher(None, t_next, segment[i:i+len(t_next)].strip()).ratio() if t_next else 0
        
        # Average the scores. Higher score = higher probability of a correct break point.
        score = (s_left + s_right) / 2
        if score > max_score:
            max_score, best_pos = score, i
            
    return best_pos, max_score

def run():
    # --- 1. CLI Argument Handling ---
    if len(sys.argv) < 3:
        print("USAGE: python srt_gen.py <audio_file> <script_file>")
        sys.exit(1)

    audio_in = sys.argv[1]
    script_arg = sys.argv[2]

    # Validate existence of the input audio
    if not os.path.exists(audio_in):
        print(f"ERROR: Audio file '{audio_in}' not found.")
        sys.exit(1)

    # Validate existence of the script (supports .txt suffix omission)
    script_in = script_arg
    if not os.path.exists(script_in):
        if os.path.exists(script_arg + ".txt"):
            script_in = script_arg + ".txt"
        else:
            print(f"ERROR: Script '{script_arg}' not found.")
            sys.exit(1)

    # Define output structure based on script filename
    audio_name = os.path.splitext(os.path.basename(audio_in))[0]
    script_name = os.path.splitext(os.path.basename(script_in))[0]
    output_dir = script_name
    output_srt = audio_name + ".srt"

    # --- 2. Audio Standardization ---
    # Whisper AI performs best with 16kHz mono WAV. 
    # We pre-convert non-WAV files to a standardized temporary WAV.
    current_audio_workfile = audio_in
    temp_wav_file = None
    if not audio_in.lower().endswith(".wav"):
        print(f"NOTICE: Pre-processing {audio_in} to standardized WAV...")
        fd, temp_wav_file = tempfile.mkstemp(suffix=".wav")
        os.close(fd) # File descriptor is not needed for ffmpeg call
        conv_cmd = ["ffmpeg", "-y", "-i", audio_in, "-ac", "1", "-ar", "16000", temp_wav_file]
        subprocess.run(conv_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        current_audio_workfile = temp_wav_file

    try:
        # Create directory for script fragment storage (debugging/archiving)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Load master script and normalize whitespace
        with open(script_in, "r", encoding="utf-8") as f:
            master_script = " ".join(f.read().split())

        # --- 3. Silence Detection Phase ---
        print(f"STEP 1: Scanning {audio_in} for speech boundaries...")
        # We use ffmpeg's silencedetect filter. Parameters:
        # noise=-30dB: volume threshold for silence
        # d=0.5: minimum duration of silence (sec)
        log_cmd = ["ffmpeg", "-i", current_audio_workfile, "-af", "silencedetect=noise=-30dB:d=0.5", "-f", "null", "-"]
        
        # FIX: UnicodeDecodeError handling.
        # FFmpeg on Windows may output using system locales (e.g. CP932/1252) which conflict with UTF-8.
        # 'errors="replace"' ensures the script extracts timestamps even if metadata contains garbled characters.
        result = subprocess.run(log_cmd, stderr=subprocess.PIPE, text=True, errors="replace")
        
        # Parse silence intervals from ffmpeg's stderr output
        silence_starts = re.findall(r"silence_start: ([\d\.]+)", result.stderr)
        silence_ends = re.findall(r"silence_end: ([\d\.]+)", result.stderr)

        # Map non-silent intervals as "Speech Segments"
        segments = []
        for i in range(len(silence_ends)):
            start = float(silence_ends[i])
            end = float(silence_starts[i+1]) if i + 1 < len(silence_starts) else start + 5.0
            if end - start > 0.1: # Skip segments too short to contain speech
                segments.append({"start": start, "end": end, "dur": end - start})

        print(f"Total segments detected: {len(segments)}")
        
        ai_model = None
        remaining_script = master_script
        srt_data = []

        # --- 4. Alignment & Transcription Loop ---
        for i, seg in enumerate(segments):
            timestamp_ms = int(seg['start'] * 1000)
            txt_file_path = os.path.join(output_dir, f"{timestamp_ms:09d}.txt")
            
            final_segment_text = ""
            # RESUME LOGIC: Check if this fragment was already processed to save time/compute
            if os.path.exists(txt_file_path):
                with open(txt_file_path, "r", encoding="utf-8") as f:
                    final_segment_text = f.read().strip()
                print(f"[{i+1:03d}] SKIP: {os.path.basename(txt_file_path)} (existing fragment)")
                
                # Advance the master script pointer past the skipped segment
                find_idx = remaining_script.find(final_segment_text)
                if find_idx != -1: 
                    remaining_script = remaining_script[find_idx + len(final_segment_text):].strip()
            else:
                # Lazy-load Whisper model only when actual transcription is needed
                if ai_model is None:
                    print("STEP 2: Loading Whisper AI model (Turbo)...")
                    ai_model = whisper.load_model("turbo")

                # FIX: TypeError: expected np.ndarray (got _io.BytesIO)
                # Whisper's transcribe() is not compatible with in-memory Byte streams for audio.
                # We use mkstemp to create a physical temporary file for stable cross-platform access.
                fd1, tmp_curr = tempfile.mkstemp(suffix=".wav")
                os.close(fd1)
                subprocess.run(["ffmpeg", "-y", "-ss", str(seg['start']), "-t", str(seg['dur']), "-i", current_audio_workfile, "-ar", "16000", "-ac", "1", tmp_curr], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                transcript_curr = ai_model.transcribe(tmp_curr)["text"].strip()
                os.remove(tmp_curr) # Cleanup audio chunk immediately after transcription

                # LOOK-AHEAD: Process next segment to determine where 'current' text ends
                if i + 1 < len(segments):
                    next_seg = segments[i+1]
                    fd2, tmp_next = tempfile.mkstemp(suffix=".wav")
                    os.close(fd2)
                    subprocess.run(["ffmpeg", "-y", "-ss", str(next_seg['start']), "-t", str(next_seg['dur']), "-i", current_audio_workfile, "-ar", "16000", "-ac", "1", tmp_next], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    transcript_next = ai_model.transcribe(tmp_next)["text"].strip()
                    os.remove(tmp_next) 
                    
                    # Align transcript with the script to find the precise character index for the split
                    split_idx, match_score = get_refined_split_pos(transcript_curr, transcript_next, remaining_script, 45, 400, 5.5, seg['dur'])
                else:
                    # Final segment: take everything remaining in the script
                    split_idx = len(remaining_script)
                    match_score = 1.0

                # Extract the final aligned text fragment
                final_segment_text = remaining_script[:split_idx].strip()
                with open(txt_file_path, "w", encoding="utf-8") as f:
                    f.write(final_segment_text)
                
                # Move pointer to the next part of the script
                remaining_script = remaining_script[split_idx:].strip()

                # LOGGING: Detailed progress with text preview for developer monitoring
                preview = final_segment_text.replace("\n", " ")
                preview = (preview[:37] + "...") if len(preview) > 40 else preview
                print(f"[{i+1:03d}] SAVED: {os.path.basename(txt_file_path)} -> \"{preview}\" (Score: {match_score:.2f})")

            # Store formatted SRT block
            start_time_srt = format_srt_time(seg['start'])
            end_time_srt = format_srt_time(seg['end'])
            srt_data.append(f"{i+1}\n{start_time_srt} --> {end_time_srt}\n{final_segment_text}\n")

        # --- 5. Export ---
        with open(output_srt, "w", encoding="utf-8") as f:
            f.write("\n".join(srt_data))
        print(f"\nSUCCESS:\n- Text fragments: {output_dir}/\n- Final Subtitles: {output_srt}")

    finally:
        # --- 6. Final Cleanup ---
        # Ensure the master workfile is removed even if the script crashes
        if temp_wav_file and os.path.exists(temp_wav_file):
            os.remove(temp_wav_file)
            print("CLEANUP: Master temporary audio file removed.")

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(0)

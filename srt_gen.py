import os
import subprocess
import re
import whisper
import sys
import tempfile
from difflib import SequenceMatcher
from datetime import timedelta

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Minimum similarity score (0.0 to 1.0) required to accept a match.
# If below this, the segment is treated as noise/ad-lib and skipped.
SCORE_THRESHOLD = 0.5

def format_srt_time(seconds):
    """
    Converts float seconds into standard SRT timestamp format: HH:MM:SS,mmm.
    """
    td = timedelta(seconds=seconds)
    total_sec = int(td.total_seconds())
    msec = int(td.microseconds / 1000)
    return f"{total_sec//3600:02}:{(total_sec%3600)//60:02}:{total_sec%60:02},{msec:03}"

def get_refined_split_pos(curr_heard, next_heard, script_segment, dur, was_skip):
    """
    PHONETIC ANCHOR ALGORITHM (High-Precision Legacy Logic):
    Calculates the optimal split point by comparing audio transcriptions 
    with precise slices of the master script.
    """
    # APPLY DYNAMIC WINDOW RULE:
    # Use 350 chars if the audio is long (>=5.5s) OR recovering from a SKIP.
    # Otherwise, use a narrow 45-char window to prevent "drifting" errors.
    limit = 350 if (dur >= 5.5 or was_skip) else 45
    segment = script_segment[:limit]
    
    # Extract phonetic anchors from Whisper results.
    # Right Anchor: First 2 words of the next segment.
    # Left Anchor: Last 2 words of the current segment.
    words_next = next_heard.split()[:2]
    words_curr = curr_heard.split()[-2:]
    t_next, t_curr = " ".join(words_next), " ".join(words_curr)
    
    best_pos, max_score = 0, -1
    
    # Iterate through every character position in the search window.
    for i in range(len(segment) + 1):
        # CORE PRECISION LOGIC: 
        # We slice the script at index 'i' and compare it to the anchors.
        # .strip() ensures that whitespace doesn't dilute the similarity score.
        
        # Match script's left-of-cut with current audio's tail.
        s_left = SequenceMatcher(None, t_curr, segment[max(0, i-len(t_curr)):i].strip()).ratio() if t_curr else 1.0
        # Match script's right-of-cut with next audio's head.
        s_right = SequenceMatcher(None, t_next, segment[i:i+len(t_next)].strip()).ratio() if t_next else 1.0
        
        # Calculate the mean probability.
        score = (s_left + s_right) / 2
        if score > max_score:
            max_score, best_pos = score, i
            
    return best_pos, max_score

def run():
    # Ensure mandatory file arguments are present.
    if len(sys.argv) < 3:
        print("USAGE: python srt_gen.py <audio_file> <script_file>")
        sys.exit(1)

    audio_in, script_arg = sys.argv[1], sys.argv[2]
    script_in = script_arg if os.path.exists(script_arg) else script_arg + ".txt"
    output_dir = os.path.splitext(os.path.basename(script_in))[0]
    output_srt = os.path.splitext(os.path.basename(audio_in))[0] + ".srt"

    # PRE-PROCESSING: Normalize audio to 16kHz Mono WAV for Whisper AI.
    current_audio_workfile = audio_in
    temp_wav_file = None
    if not audio_in.lower().endswith(".wav"):
        fd, temp_wav_file = tempfile.mkstemp(suffix=".wav"); os.close(fd)
        subprocess.run(["ffmpeg", "-y", "-i", audio_in, "-ac", "1", "-ar", "16000", temp_wav_file], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        current_audio_workfile = temp_wav_file

    try:
        # Load the master script and normalize all whitespace into single spaces.
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        with open(script_in, "r", encoding="utf-8") as f:
            master_script = " ".join(f.read().split())

        # VAD (Voice Activity Detection): Map speech timestamps using FFmpeg's silencedetect.
        log_cmd = ["ffmpeg", "-i", current_audio_workfile, "-af", "silencedetect=noise=-30dB:d=0.5", "-f", "null", "-"]
        result = subprocess.run(log_cmd, stderr=subprocess.PIPE, text=True, errors="replace")
        s_starts = re.findall(r"silence_start: ([\d\.]+)", result.stderr)
        s_ends = re.findall(r"silence_end: ([\d\.]+)", result.stderr)

        # Build segment list based on non-silent durations.
        segments = []
        for i in range(len(s_ends)):
            start = float(s_ends[i])
            end = float(s_starts[i+1]) if i+1 < len(s_starts) else start + 5.0
            if end - start > 0.1: 
                segments.append({"start": start, "end": end, "dur": end - start})

        # --- STATE MANAGEMENT (Coordinate-based) ---
        # last_cut_pos: Absolute character index on the script "rope".
        # last_good_text: Cached transcription of the last successful match.
        # was_skip: Flag to trigger the 350-char window exception.
        last_cut_pos = 0     
        last_good_text = ""  
        was_skip = False     
        
        srt_data, ai_model = [], None

        for i, seg in enumerate(segments):
            curr_num = i + 1
            ts_ms = int(seg['start'] * 1000)
            txt_path = os.path.join(output_dir, f"{ts_ms:09d}.txt")

            # A. RESUME LOGIC: Check if a valid .txt file already exists on disk.
            if os.path.exists(txt_path) and os.path.getsize(txt_path) > 0:
                with open(txt_path, "r", encoding="utf-8") as f:
                    final_text = f.read()
                # Fast-forward the script pointer to the end of the cached text.
                idx = master_script.find(final_text, last_cut_pos)
                if idx != -1:
                    last_cut_pos = idx + len(final_text)
                    last_good_text = final_text
                score = 1.0
                was_skip = False
            else:
                # B. PHYSICAL CLEANUP: Ensure skipped segments do NOT leave files behind.
                if os.path.exists(txt_path): os.remove(txt_path)

                # Initialize AI model only when needed.
                if ai_model is None: ai_model = whisper.load_model("turbo")

                def get_trans(s, d):
                    """Extracts audio chunk and transcribes it via Whisper."""
                    fd, tmp = tempfile.mkstemp(suffix=".wav"); os.close(fd)
                    subprocess.run(["ffmpeg", "-y", "-ss", str(s), "-t", str(d), "-i", current_audio_workfile, 
                                    "-ar", "16000", "-ac", "1", tmp], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    txt = ai_model.transcribe(tmp)["text"].strip(); os.remove(tmp); return txt

                # Perform double-transcription to find the precise boundary.
                t_curr = get_trans(seg['start'], seg['dur'])
                t_next = get_trans(segments[i+1]['start'], segments[i+1]['dur']) if i+1 < len(segments) else ""
                
                # The 'Left Anchor' is the tail of the last verified audio to prevent drift.
                left_anchor = last_good_text if last_good_text else t_curr
                search_window = master_script[last_cut_pos : last_cut_pos + 1000]

                # C. ALIGNMENT CALCULATION: Apply the 45/350 window rule.
                rel_split, score = get_refined_split_pos(t_curr, t_next, search_window, seg['dur'], was_skip) \
                                   if i+1 < len(segments) else (len(search_window), 1.0)

                if score >= SCORE_THRESHOLD:
                    # SUCCESS: Capture script text from last_cut_pos to the new split point.
                    # This automatically includes any text belonging to previously skipped segments.
                    final_text = master_script[last_cut_pos : last_cut_pos + rel_split].strip()
                    
                    # Update coordinate and anchor for the next segment.
                    last_cut_pos += rel_split
                    last_good_text = t_curr 
                    was_skip = False
                    
                    # Persist success to disk.
                    with open(txt_path, "w", encoding="utf-8") as f: f.write(final_text)
                else:
                    # SKIP: Score is too low. Freeze the pointer and do not create a file.
                    final_text = ""
                    was_skip = True

            # MONITORING: Output progress and the search window size used.
            label = "350w" if (seg['dur'] >= 5.5 or was_skip) else "45w"
            print(f"[{curr_num:04d}] (Score:{score:.2f}) {final_text[:30] if final_text else 'SKIP'}")
            
            # Populate SRT buffer (Skips will result in empty subtitle text).
            srt_data.append(f"{curr_num}\n{format_srt_time(seg['start'])} --> {format_srt_time(seg['end'])}\n{final_text}\n")

        # Save final SRT file.
        with open(output_srt, "w", encoding="utf-8") as f: f.write("\n".join(srt_data))
        
    finally:
        # Cleanup temporary WAV artifacts.
        if temp_wav_file and os.path.exists(temp_wav_file): os.remove(temp_wav_file)

if __name__ == "__main__":
    run()

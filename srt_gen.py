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

# Minimum confidence required to consume the script. Prevents cascading sync errors.
SCORE_THRESHOLD = 0.5

# ==============================================================================
# UTILITY: SRT TIMESTAMP GENERATOR
# ==============================================================================

def format_srt_time(seconds):
    """
    PURPOSE: Converts raw float seconds into HH:MM:SS,mmm format.
    LOGIC: Formats milliseconds with a comma for SRT standard compatibility.
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
    PURPOSE: Finds the mathematically optimal split index in the script text.
    LOGIC: Matches trailing audio anchors and leading next-segment anchors against script.
    """
    words_next = next_heard.split()[:2]
    words_curr = curr_heard.split()[-2:]
    t_next = " ".join(words_next)
    t_curr = " ".join(words_curr)
    
    limit = extended_limit if dur >= long_wav_sec else base_limit
    segment = script_segment[:limit]
    
    best_pos, max_score = 0, -1
    for i in range(len(segment) + 1):
        s_left = SequenceMatcher(None, t_curr, segment[max(0, i-len(t_curr)):i].strip()).ratio() if t_curr else 0
        s_right = SequenceMatcher(None, t_next, segment[i:i+len(t_next)].strip()).ratio() if t_next else 0
        
        score = (s_left + s_right) / 2
        if score > max_score:
            max_score, best_pos = score, i
            
    return best_pos, max_score

# ==============================================================================
# MAIN PROCESSING ENGINE
# ==============================================================================

def run():
    # --- STEP 1: PATH RESOLUTION & ARGUMENT VALIDATION ---
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

    audio_name = os.path.splitext(os.path.basename(audio_in))[0]
    script_name = os.path.splitext(os.path.basename(script_in))[0]
    output_dir = script_name
    output_srt = audio_name + ".srt"

    # --- STEP 2: AUDIO SIGNAL STANDARDIZATION ---
    current_audio_workfile = audio_in
    temp_wav_file = None
    if not audio_in.lower().endswith(".wav"):
        print(f"NOTICE: Creating standardized 16kHz Mono WAV workfile...")
        fd, temp_wav_file = tempfile.mkstemp(suffix=".wav")
        os.close(fd) 
        conv_cmd = ["ffmpeg", "-y", "-i", audio_in, "-ac", "1", "-ar", "16000", temp_wav_file]
        subprocess.run(conv_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        current_audio_workfile = temp_wav_file

    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(script_in, "r", encoding="utf-8") as f:
            master_script = " ".join(f.read().split())

        # --- STEP 3: VAD (VOICE ACTIVITY DETECTION) ---
        print(f"STEP 1: Mapping speech intervals via FFmpeg silencedetect...")
        log_cmd = ["ffmpeg", "-i", current_audio_workfile, "-af", "silencedetect=noise=-30dB:d=0.5", "-f", "null", "-"]
        result = subprocess.run(log_cmd, stderr=subprocess.PIPE, text=True, errors="replace")
        
        silence_starts = re.findall(r"silence_start: ([\d\.]+)", result.stderr)
        silence_ends = re.findall(r"silence_end: ([\d\.]+)", result.stderr)

        segments = []
        for i in range(len(silence_ends)):
            start = float(silence_ends[i])
            end = float(silence_starts[i+1]) if i + 1 < len(silence_starts) else start + 5.0
            if end - start > 0.1:
                segments.append({"start": start, "end": end, "dur": end - start})

        total_count = len(segments)
        print(f"Total speech segments detected: {total_count}")
        
        ai_model = None
        remaining_script = master_script
        srt_data = []

        # --- STEP 4: TRANSCRIPTION & ALIGNMENT LOOP ---
        for i, seg in enumerate(segments):
            curr_num = i + 1
            timestamp_ms = int(seg['start'] * 1000)
            txt_filename = f"{timestamp_ms:09d}.txt"
            txt_file_path = os.path.join(output_dir, txt_filename)
            
            final_segment_text = ""
            match_score = 0.0 

            if os.path.exists(txt_file_path):
                with open(txt_file_path, "r", encoding="utf-8") as f:
                    final_segment_text = f.read()
                if final_segment_text:
                    find_idx = remaining_script.find(final_segment_text)
                    if find_idx != -1: 
                        remaining_script = remaining_script[find_idx + len(final_segment_text):].strip()
            else:
                if ai_model is None:
                    print("STEP 2: Initializing Whisper AI (Turbo model)...")
                    ai_model = whisper.load_model("turbo")

                fd1, tmp_curr = tempfile.mkstemp(suffix=".wav")
                os.close(fd1)
                subprocess.run(["ffmpeg", "-y", "-ss", str(seg['start']), "-t", str(seg['dur']), "-i", current_audio_workfile, "-ar", "16000", "-ac", "1", tmp_curr], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                transcript_curr = ai_model.transcribe(tmp_curr)["text"].strip()
                os.remove(tmp_curr) 

                if curr_num < total_count:
                    next_seg = segments[i+1]
                    fd2, tmp_next = tempfile.mkstemp(suffix=".wav")
                    os.close(fd2)
                    subprocess.run(["ffmpeg", "-y", "-ss", str(next_seg['start']), "-t", str(next_seg['dur']), "-i", current_audio_workfile, "-ar", "16000", "-ac", "1", tmp_next], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    transcript_next = ai_model.transcribe(tmp_next)["text"].strip()
                    os.remove(tmp_next) 
                    
                    split_idx, match_score = get_refined_split_pos(transcript_curr, transcript_next, remaining_script, 45, 400, 5.5, seg['dur'])
                    
                    # --- RETROACTIVE BACKTRACKING LOGIC ---
                    # Theoretical implementation to fill previously skipped gaps.
                    t_curr_end = " ".join(transcript_curr.split()[-2:])
                    if match_score >= SCORE_THRESHOLD and t_curr_end:
                        search_limit = 400 if seg['dur'] >= 5.5 else 45
                        found_pos = remaining_script[:search_limit].rfind(t_curr_end)
                        if found_pos != -1:
                            new_end_pos = found_pos + len(t_curr_end)
                            if new_end_pos > split_idx:
                                missed_text = remaining_script[split_idx:new_end_pos].strip()
                                # Patch the previous block in memory and on disk.
                                if i > 0:
                                    prev_lines = srt_data[i-1].split('\n')
                                    prev_lines[2] = (prev_lines[2] + " " + missed_text).strip()
                                    srt_data[i-1] = "\n".join(prev_lines)
                                    prev_ts = int(segments[i-1]['start'] * 1000)
                                    with open(os.path.join(output_dir, f"{prev_ts:09d}.txt"), "a", encoding="utf-8") as f:
                                        f.write(" " + missed_text)
                                split_idx = new_end_pos
                else:
                    split_idx = len(remaining_script)
                    match_score = 1.0

                if match_score >= SCORE_THRESHOLD:
                    final_segment_text = remaining_script[:split_idx].strip()
                    actual_consumed = split_idx
                else:
                    final_segment_text = ""
                    actual_consumed = 0
                
                with open(txt_file_path, "w", encoding="utf-8", newline='') as f:
                    f.write(final_segment_text)
                
                remaining_script = remaining_script[actual_consumed:].strip()

            # --- MONITORING: CONSOLE FEEDBACK (EXACT DESIGN RESTORED) ---
            preview = final_segment_text.replace("\n", " ")
            preview = (preview[:37] + "...") if len(preview) > 40 else preview
            log_msg = f"[{curr_num:04d}/{total_count:04d}] (Score:{match_score:.2f}) "
            print(f"{log_msg}{preview}" if final_segment_text else f"{log_msg}SKIP (Pointer Held)")

            start_time_srt = format_srt_time(seg['start'])
            end_time_srt = format_srt_time(seg['end'])
            srt_data.append(f"{curr_num}\n{start_time_srt} --> {end_time_srt}\n{final_segment_text}\n")

        # --- STEP 5: FINAL SRT ASSEMBLY ---
        with open(output_srt, "w", encoding="utf-8") as f:
            f.write("\n".join(srt_data))
        print(f"\nFINISH:\n- Fragments saved in: {output_dir}/\n- Master Subtitles: {output_srt}")

    finally:
        # --- STEP 6: CLEANUP ---
        if temp_wav_file and os.path.exists(temp_wav_file):
            os.remove(temp_wav_file)
            print("CLEANUP: Process workfile removed.")

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\nProcess aborted by user.")
        sys.exit(0)

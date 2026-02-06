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
# CONFIGURATION
# ==============================================================================
SCORE_THRESHOLD = 0.5

def format_srt_time(seconds):
    """PURPOSE: Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    td = timedelta(seconds=seconds)
    total_sec = int(td.total_seconds())
    msec = int(td.microseconds / 1000)
    return f"{total_sec//3600:02}:{(total_sec%3600)//60:02}:{total_sec%60:02},{msec:03}"

def get_refined_split_pos(curr_heard, next_heard, script_segment, base_limit, extended_limit, long_wav_sec, dur):
    """
    PURPOSE: Finds the optimal split index by comparing the 'tail' of current 
             audio and 'head' of next audio against the script.
    """
    words_next, words_curr = next_heard.split()[:2], curr_heard.split()[-2:]
    t_next, t_curr = " ".join(words_next), " ".join(words_curr)
    
    limit = extended_limit if dur >= long_wav_sec else base_limit
    segment = script_segment[:limit]
    
    best_pos, max_score = 0, -1
    for i in range(len(segment) + 1):
        # The core comparison: Current tail vs script-left, Next head vs script-right
        s_left = SequenceMatcher(None, t_curr, segment[max(0, i-len(t_curr)):i].strip()).ratio() if t_curr else 0
        s_right = SequenceMatcher(None, t_next, segment[i:i+len(t_next)].strip()).ratio() if t_next else 0
        score = (s_left + s_right) / 2
        if score > max_score:
            max_score, best_pos = score, i
    return best_pos, max_score

def run():
    # --- STEP 1: INITIALIZATION ---
    if len(sys.argv) < 3:
        print("USAGE: python srt_gen.py <audio_file> <script_file>")
        sys.exit(1)

    audio_in, script_arg = sys.argv[1], sys.argv[2]
    script_in = script_arg if os.path.exists(script_arg) else script_arg + ".txt"

    audio_name = os.path.splitext(os.path.basename(audio_in))[0]
    output_dir = os.path.splitext(os.path.basename(script_in))[0]
    output_srt = audio_name + ".srt"

    # --- STEP 2: AUDIO STANDARDIZATION ---
    current_audio_workfile = audio_in
    temp_wav_file = None
    if not audio_in.lower().endswith(".wav"):
        fd, temp_wav_file = tempfile.mkstemp(suffix=".wav"); os.close(fd)
        subprocess.run(["ffmpeg", "-y", "-i", audio_in, "-ac", "1", "-ar", "16000", temp_wav_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        current_audio_workfile = temp_wav_file

    try:
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        with open(script_in, "r", encoding="utf-8") as f:
            master_script = " ".join(f.read().split())

        # --- STEP 3: VAD ---
        print(f"STEP 1: Mapping speech intervals via FFmpeg silencedetect...")
        log_cmd = ["ffmpeg", "-i", current_audio_workfile, "-af", "silencedetect=noise=-30dB:d=0.5", "-f", "null", "-"]
        result = subprocess.run(log_cmd, stderr=subprocess.PIPE, text=True, errors="replace")
        silence_starts, silence_ends = re.findall(r"silence_start: ([\d\.]+)", result.stderr), re.findall(r"silence_end: ([\d\.]+)", result.stderr)

        segments = []
        for i in range(len(silence_ends)):
            start = float(silence_ends[i])
            end = float(silence_starts[i+1]) if i + 1 < len(silence_starts) else start + 5.0
            if end - start > 0.1: segments.append({"start": start, "end": end, "dur": end - start})

        total_count, ai_model, remaining_script, srt_data = len(segments), None, master_script, []
        print(f"Total speech segments detected: {total_count}")

        # --- STEP 4: ALIGNMENT LOOP ---
        for i, seg in enumerate(segments):
            curr_num = i + 1
            timestamp_ms = int(seg['start'] * 1000)
            txt_file_path = os.path.join(output_dir, f"{timestamp_ms:09d}.txt")
            final_segment_text, match_score = "", 0.0 

            if os.path.exists(txt_file_path):
                with open(txt_file_path, "r", encoding="utf-8") as f: final_segment_text = f.read()
                if final_segment_text:
                    idx = remaining_script.find(final_segment_text)
                    if idx != -1: remaining_script = remaining_script[idx + len(final_segment_text):].strip()
            else:
                if ai_model is None:
                    print("STEP 2: Initializing Whisper AI (Turbo model)...")
                    ai_model = whisper.load_model("turbo")

                def get_trans(s, d):
                    fd, tmp = tempfile.mkstemp(suffix=".wav"); os.close(fd)
                    subprocess.run(["ffmpeg", "-y", "-ss", str(s), "-t", str(d), "-i", current_audio_workfile, "-ar", "16000", "-ac", "1", tmp], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    txt = ai_model.transcribe(tmp)["text"].strip(); os.remove(tmp); return txt

                t_curr = get_trans(seg['start'], seg['dur'])
                t_next = get_trans(segments[i+1]['start'], segments[i+1]['dur']) if curr_num < total_count else ""
                
                # RE-SYNC: Use the original "Tail-to-Head" comparison to find the split
                split_idx, match_score = get_refined_split_pos(t_curr, t_next, remaining_script, 45, 400, 5.5, seg['dur']) if curr_num < total_count else (len(remaining_script), 1.0)

                if match_score >= SCORE_THRESHOLD:
                    # Everything up to split_idx is consumed from the script
                    all_consumed = remaining_script[:split_idx].strip()
                    
                    # LOGIC: Current text is the core; any "leading garbage" is pushed back
                    # To keep it simple and accurate, we use the found split_idx directly
                    final_segment_text = all_consumed
                    
                    # ADVANCE POINTER: No characters are left behind in the master script
                    remaining_script = remaining_script[split_idx:].strip()
                else:
                    final_segment_text = ""

                with open(txt_file_path, "w", encoding="utf-8") as f: f.write(final_segment_text)

            # --- MONITORING ---
            preview = (final_segment_text[:37] + "...") if len(final_segment_text) > 40 else final_segment_text
            log_msg = f"[{curr_num:04d}/{total_count:04d}] (Score:{match_score:.2f}) "
            print(f"{log_msg}{preview}" if final_segment_text else f"{log_msg}SKIP (Pointer Held)")

            srt_data.append(f"{curr_num}\n{format_srt_time(seg['start'])} --> {format_srt_time(seg['end'])}\n{final_segment_text}\n")

        with open(output_srt, "w", encoding="utf-8") as f: f.write("\n".join(srt_data))
        print(f"\nFINISH: {output_srt}")
    finally:
        if temp_wav_file and os.path.exists(temp_wav_file): os.remove(temp_wav_file)

if __name__ == "__main__":
    run()

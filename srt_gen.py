import os, subprocess, re, whisper, sys, tempfile
from difflib import SequenceMatcher
from datetime import timedelta

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Minimum similarity ratio (0.0 to 1.0) to accept a script match.
SCORE_THRESHOLD = 0.5

def format_srt_time(seconds):
    """
    Standard utility to convert float seconds into SRT timestamp format: HH:MM:SS,mmm.
    """
    td = timedelta(seconds=seconds)
    total_sec = int(td.total_seconds())
    msec = int(td.microseconds / 1000)
    return f"{total_sec//3600:02}:{(total_sec%3600)//60:02}:{total_sec%60:02},{msec:03}"

def get_refined_split_pos(curr_heard, next_heard, script_segment, dur):
    """
    LEGACY PRECISION FORMULA:
    Calculates the optimal character split point 'i' by anchoring 
    the left side to the CURRENT audio and the right side to the NEXT audio.
    """
    # Use a larger search window (350 chars) for longer audio or recovery.
    limit = 350 if dur >= 5.5 else 45
    segment = script_segment[:limit]
    
    # Phonetic Anchors: Extract start/end words to focus the SequenceMatcher.
    words_next = next_heard.split()[:2]
    words_curr = curr_heard.split()[-2:]
    t_next, t_curr = " ".join(words_next), " ".join(words_curr)
    
    best_pos, max_score = 0, -1
    
    # Iterate through the segment to find the point where both anchors match best.
    for i in range(len(segment) + 1):
        # Match script's tail (left of 'i') with current audio's tail.
        s_left = SequenceMatcher(None, t_curr, segment[max(0, i-len(t_curr)):i].strip()).ratio() if t_curr else 1.0
        # Match script's head (right of 'i') with next audio's head.
        s_right = SequenceMatcher(None, t_next, segment[i:i+len(t_next)].strip()).ratio() if t_next else 1.0
        
        # Mean average of both directions determines the final score.
        score = (s_left + s_right) / 2
        if score > max_score:
            max_score, best_pos = score, i
            
    return best_pos, max_score

def run():

    if len(sys.argv) < 3:
        print("USAGE: python srt_gen.py <audio_file> <script_file>")
        sys.exit(1)

    print("Initializing Whisper...")

    audio_in, script_arg = sys.argv[1], sys.argv[2]
    script_in = script_arg if os.path.exists(script_arg) else script_arg + ".txt"
    output_dir = os.path.splitext(os.path.basename(script_in))[0]
    output_srt = os.path.splitext(os.path.basename(audio_in))[0] + ".srt"

    # AUDIO PRE-PROCESSING:
    # Convert input to 16kHz mono WAV to ensure maximum Whisper accuracy.
    current_audio_workfile = audio_in
    temp_wav_file = None
    if not audio_in.lower().endswith(".wav"):
        fd, temp_wav_file = tempfile.mkstemp(suffix=".wav"); os.close(fd)
        subprocess.run(["ffmpeg", "-y", "-i", audio_in, "-ac", "1", "-ar", "16000", temp_wav_file], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        current_audio_workfile = temp_wav_file

    try:
        # Load and clean the master script.
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        with open(script_in, "r", encoding="utf-8") as f:
            master_script = " ".join(f.read().split())

        # VAD (Voice Activity Detection):
        # Analyze the audio for silence to determine segment boundaries.
        log_cmd = ["ffmpeg", "-i", current_audio_workfile, "-af", "silencedetect=noise=-30dB:d=0.5", "-f", "null", "-"]
        result = subprocess.run(log_cmd, stderr=subprocess.PIPE, text=True, errors="replace")
        s_starts = re.findall(r"silence_start: ([\d\.]+)", result.stderr)
        s_ends = re.findall(r"silence_end: ([\d\.]+)", result.stderr)

        segments = []
        for i in range(len(s_ends)):
            start = float(s_ends[i])
            end = float(s_starts[i+1]) if i+1 < len(s_starts) else start + 5.0
            if end - start > 0.1: 
                segments.append({"start": start, "end": end, "dur": end - start})

        # --- STATE MANAGEMENT ---
        last_cut_pos = 0     # The absolute character index in the master script.
        success_history = [] # Stack of successful segments for recursive boundary correction.
        total_segs = len(segments)
        ai_model = None      # Lazy-load Whisper model only when transcription starts.

        def get_trans(s, d):
            """Internal transcription helper using a temporary audio slice."""
            nonlocal ai_model
            if ai_model is None:
                ai_model = whisper.load_model("turbo")
            fd, tmp = tempfile.mkstemp(suffix=".wav"); os.close(fd)
            subprocess.run(["ffmpeg", "-y", "-ss", str(s), "-t", str(d), "-i", current_audio_workfile, 
                            "-ar", "16000", "-ac", "1", tmp], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            txt = ai_model.transcribe(tmp)["text"].strip(); os.remove(tmp); return txt

        # --- RETROSPECTIVE SLIDING LOOP ---
        i = 0
        while i < len(segments):
            seg = segments[i]
            # Transcribe current audio chunk.
            t_curr = get_trans(seg['start'], seg['dur'])
            # Peek at the next chunk to act as a 'Right Anchor' for the current segment.
            t_next_tentative = get_trans(segments[i+1]['start'], segments[i+1]['dur']) if i+1 < len(segments) else ""
            
            # Slice the script starting from our last confirmed position.
            search_window = master_script[last_cut_pos : last_cut_pos + 1000]
            
            # RUN THE TEST:
            rel_split, score = get_refined_split_pos(t_curr, t_next_tentative, search_window, seg['dur'])

            display_text = ""
            if score >= SCORE_THRESHOLD:
                # SUCCESS: This audio segment is valid.
                # CORE LOGIC: If we have a previous segment (e.g. 117), its boundary 
                # might have been guessed using 'garbage' (e.g. 118). 
                # We now re-calculate 117's end using the current VALID audio (t_curr).
                if success_history:
                    prev = success_history[-1]
                    prev_window = master_script[prev['start_pos'] : prev['start_pos'] + 1000]
                    
                    # RE-CALCULATE previous segment's end with the new Right Anchor (t_curr).
                    refined_pos, _ = get_refined_split_pos(prev['trans'], t_curr, prev_window, prev['dur'])
                    
                    # Overwrite the previous segment's text with the corrected slice.
                    prev_text = prev_window[:refined_pos].strip()
                    with open(prev['path'], "w", encoding="utf-8") as f: f.write(prev_text)
                    
                    # Sync the global script pointer to the corrected boundary.
                    last_cut_pos = prev['start_pos'] + refined_pos

                # Save the current segment.
                ts_ms = int(seg['start'] * 1000)
                txt_path = os.path.join(output_dir, f"{ts_ms:09d}.txt")
                current_window = master_script[last_cut_pos : last_cut_pos + 1000]
                final_text = current_window[:rel_split].strip()
                with open(txt_path, "w", encoding="utf-8") as f: f.write(final_text)

                # Record the success. This entry will be 're-calculated' by the next OK segment.
                success_history.append({
                    'path': txt_path, 'start_pos': last_cut_pos, 'trans': t_curr, 
                    'dur': seg['dur'], 'start_time': seg['start'], 'end_time': seg['end']
                })
                
                # Advance the pointer tentatively.
                last_cut_pos += rel_split
                display_text = final_text
                i += 1
            else:
                # FAILURE: This audio is garbage.
                # We move to the next audio segment (i), but we DO NOT move last_cut_pos.
                # The next audio chunk will attempt to align itself starting from the end of the last GOOD segment.
                display_text = "SKIP"
                i += 1 

            # Progress Reporting:
            print(f"[{i:04d}/{total_segs:04d}] (Score:{score:.2f}) {display_text[:50]}")

        # --- SRT ASSEMBLY ---
        # Construct the final SRT file using the corrected text files.
        srt_final = []
        for idx, h in enumerate(success_history):
            with open(h['path'], "r", encoding="utf-8") as f:
                txt = f.read()
            srt_final.append(f"{idx+1}\n{format_srt_time(h['start_time'])} --> {format_srt_time(h['end_time'])}\n{txt}\n")
        
        with open(output_srt, "w", encoding="utf-8") as f: f.write("\n".join(srt_final))

    finally:
        # Final cleanup of temporary files.
        if temp_wav_file and os.path.exists(temp_wav_file): os.remove(temp_wav_file)

if __name__ == "__main__":
    run()

import os, subprocess, re, whisper, sys, tempfile
from difflib import SequenceMatcher
from datetime import timedelta

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# The strictness of the script matching (0.0 to 1.0).
# Segments scoring below this are considered "garbage" (noise/ad-libs).
SCORE_THRESHOLD = 0.5

def format_srt_time(seconds):
    """
    Utility: Converts float seconds to standard SRT format (HH:MM:SS,mmm).
    """
    td = timedelta(seconds=seconds)
    total_sec = int(td.total_seconds())
    msec = int(td.microseconds / 1000)
    return f"{total_sec//3600:02}:{(total_sec%3600)//60:02}:{total_sec%60:02},{msec:03}"

def get_refined_split_pos(curr_heard, next_heard, script_segment, dur):
    """
    THE LEGACY PRECISION FORMULA:
    This core algorithm calculates the optimal character-level split point 'i'.
    It validates the "end" of the current audio by checking if the "start" 
    of the next audio aligns perfectly with the script immediately after 'i'.
    """
    # Sliding search window: 350 chars for long clips to ensure recovery.
    limit = 350 if dur >= 5.5 else 45
    segment = script_segment[:limit]
    
    # Anchor Extraction: Uses the start/end words as phonetic fingerprints.
    words_next = next_heard.split()[:2]
    words_curr = curr_heard.split()[-2:]
    t_next, t_curr = " ".join(words_next), " ".join(words_curr)
    
    best_pos, max_score = 0, -1
    
    # Brute-force through the window to find the point where both anchors match best.
    for i in range(len(segment) + 1):
        # s_left: Similarity of the script tail to the current audio tail.
        s_left = SequenceMatcher(None, t_curr, segment[max(0, i-len(t_curr)):i].strip()).ratio() if t_curr else 1.0
        # s_right: Similarity of the script head to the next audio head.
        s_right = SequenceMatcher(None, t_next, segment[i:i+len(t_next)].strip()).ratio() if t_next else 1.0
        
        # Mean average of bidirectional matching.
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

    # AUDIO NORMALIZATION:
    # Converting to 16kHz Mono WAV as required for stable Whisper transcription.
    current_audio_workfile = audio_in
    temp_wav_file = None
    if not audio_in.lower().endswith(".wav"):
        fd, temp_wav_file = tempfile.mkstemp(suffix=".wav"); os.close(fd)
        subprocess.run(["ffmpeg", "-y", "-i", audio_in, "-ac", "1", "-ar", "16000", temp_wav_file], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        current_audio_workfile = temp_wav_file

    try:
        # Load the master script and collapse whitespace for character-accurate slicing.
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        with open(script_in, "r", encoding="utf-8") as f:
            master_script = " ".join(f.read().split())

        # VAD (Voice Activity Detection):
        # Identifies non-silent blocks to create initial segmentation.
        log_cmd = ["ffmpeg", "-i", current_audio_workfile, "-af", "silencedetect=noise=-30dB:d=0.5", "-f", "null", "-"]
        result = subprocess.run(log_cmd, stderr=subprocess.PIPE, text=True, errors="replace")
        s_starts = re.findall(r"silence_start: ([\d\.]+)", result.stderr)
        s_ends = re.findall(r"silence_end: ([\d\.]+)", result.stderr)

        segments = []
        for j in range(len(s_ends)):
            start = float(s_ends[j])
            end = float(s_starts[j+1]) if j+1 < len(s_starts) else start + 5.0
            if end - start > 0.1: 
                segments.append({"start": start, "end": end, "dur": end - start})

        # --- STATE MANAGEMENT ---
        last_cut_pos = 0     # Global pointer to the current position in the master script.
        success_history = [] # List of successful segments used for retrospective alignment.
        total_segs = len(segments)
        ai_model = None      # Lazy-loaded model to prioritize immediate script startup.
        
        # Display Flag: Only show "(Corrected)" info if we just recovered from a garbage segment.
        skipped_since_last_success = False

        def get_trans(s, d):
            """Internal transcription helper: slices audio and calls Whisper AI."""
            nonlocal ai_model
            if ai_model is None:
                ai_model = whisper.load_model("turbo")
            fd, tmp = tempfile.mkstemp(suffix=".wav"); os.close(fd)
            subprocess.run(["ffmpeg", "-y", "-ss", str(s), "-t", str(d), "-i", current_audio_workfile, 
                            "-ar", "16000", "-ac", "1", tmp], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            txt = ai_model.transcribe(tmp)["text"].strip(); os.remove(tmp); return txt

        # --- RECURSIVE SLIDING LOOP ---
        i = 0
        while i < len(segments):
            seg = segments[i]
            # Transcribe the current audio chunk (Candidate).
            t_curr = get_trans(seg['start'], seg['dur'])
            # Peek ahead to get the 'Right Anchor' for the legacy formula.
            t_next_tentative = get_trans(segments[i+1]['start'], segments[i+1]['dur']) if i+1 < len(segments) else ""
            
            # Identify the script window for matching.
            search_window = master_script[last_cut_pos : last_cut_pos + 1000]
            
            # VALIDATION: Check alignment using the bidirectional Legacy Formula.
            rel_split, score = get_refined_split_pos(t_curr, t_next_tentative, search_window, seg['dur'])

            if score >= SCORE_THRESHOLD:
                # --- CASE: SUCCESS ---
                
                # RETROSPECTIVE OVERWRITE LOGIC:
                # If a previous segment exists, its boundary was likely calculated using garbage.
                # We now re-calculate that boundary using the current VALID audio as the true anchor.
                if success_history:
                    prev = success_history[-1]
                    prev_window = master_script[prev['start_pos'] : prev['start_pos'] + 1000]
                    
                    # Re-run Legacy Formula: Left = Prev Audio, Right = Current Valid Audio.
                    refined_pos, _ = get_refined_split_pos(prev['trans'], t_curr, prev_window, prev['dur'])
                    
                    # Overwrite the previous segment file with corrected high-precision text.
                    prev_text = prev_window[:refined_pos].strip()
                    with open(prev['path'], "w", encoding="utf-8") as f: f.write(prev_text)
                    
                    # Sync global pointer to the newly corrected boundary.
                    last_cut_pos = prev['start_pos'] + refined_pos
                    
                    # CMD OUTPUT: Only show the "Corrected" text if we are recovering from a SKIP.
                    if skipped_since_last_success:
                        print(f"[{prev['index']:04d}/{total_segs:04d}] (Corrected) {prev_text[:50]}")

                # Save the current segment to disk.
                ts_ms = int(seg['start'] * 1000)
                txt_path = os.path.join(output_dir, f"{ts_ms:09d}.txt")
                
                # Re-slice current text based on the corrected last_cut_pos.
                current_window = master_script[last_cut_pos : last_cut_pos + 1000]
                final_text = current_window[:rel_split].strip()
                with open(txt_path, "w", encoding="utf-8") as f: f.write(final_text)

                # Log success to history (this will be corrected by the NEXT successful audio).
                success_history.append({
                    'path': txt_path, 'start_pos': last_cut_pos, 'trans': t_curr, 
                    'dur': seg['dur'], 'start_time': seg['start'], 'end_time': seg['end'],
                    'index': i + 1
                })
                
                # Advance pointer and output progress.
                last_cut_pos += rel_split
                print(f"[{i+1:04d}/{total_segs:04d}] (Score:{score:.2f}) {final_text[:50]}")
                
                # Reset skip flag after a clean, successful connection.
                skipped_since_last_success = False
                i += 1
            else:
                # --- CASE: GARBAGE (SKIP) ---
                # We do not move last_cut_pos. We simply skip this audio index.
                # The next audio will attempt to align itself starting from the end of the last good segment.
                print(f"[{i+1:04d}/{total_segs:04d}] (Score:{score:.2f}) SKIP")
                skipped_since_last_success = True
                i += 1 

        # --- FINAL SRT GENERATION ---
        # Construct the final subtitle file from the corrected history.
        srt_final = []
        for idx, h in enumerate(success_history):
            with open(h['path'], "r", encoding="utf-8") as f:
                txt = f.read()
            srt_final.append(f"{idx+1}\n{format_srt_time(h['start_time'])} --> {format_srt_time(h['end_time'])}\n{txt}\n")
        
        with open(output_srt, "w", encoding="utf-8") as f: f.write("\n".join(srt_final))

    finally:
        # Cleanup temporary audio artifacts.
        if temp_wav_file and os.path.exists(temp_wav_file): os.remove(temp_wav_file)

if __name__ == "__main__":
    run()

import os, subprocess, re, whisper, sys, tempfile
from difflib import SequenceMatcher
from datetime import timedelta

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Minimum similarity score required to accept a segment matching
SCORE_THRESHOLD = 0.5

def format_srt_time(seconds):
    """
    Converts seconds (float) into SRT time format: HH:MM:SS,mmm
    """
    td = timedelta(seconds=seconds)
    total_sec = int(td.total_seconds())
    msec = int(td.microseconds / 1000)
    return f"{total_sec//3600:02}:{(total_sec%3600)//60:02}:{total_sec%60:02},{msec:03}"

def clean_text_fully(text):
    """
    Normalizes text for both Whisper output and the master script.
    - Replaces all punctuation and symbols with a single space.
    - Converts newlines, tabs, and full-width spaces into single spaces.
    - Collapses multiple consecutive spaces into one.
    """
    # Replace various symbols/punctuation with a space
    text = re.sub(r'[「」『』、。！？!?,.，．…：；:;（）【】［］\(\)\[\]★◆▲●○◎♪■□#&%ー\-\'\"‘’“”]', ' ', text)
    # Convert all whitespace characters to a single space
    text = re.sub(r'[\r\n\t　]+', ' ', text)
    # Ensure there's only a single space between words
    text = re.sub(r' +', ' ', text)
    return text.strip()

def get_refined_split_pos(curr_heard, next_heard, script_segment, dur):
    """
    Finds the optimal splitting point in the script segment based on 
    the text heard by Whisper. Uses SequenceMatcher to calculate similarity.
    """
    # Adjust search range based on audio duration to prevent over-searching
    limit = 350 if dur >= 5.5 else 45
    segment = script_segment[:limit]
    
    # Use the last two words of the current text and the first two words of the next as 'anchors'
    words_next = next_heard.split()[:2]
    words_curr = curr_heard.split()[-2:]
    t_next, t_curr = " ".join(words_next), " ".join(words_curr)
    
    best_pos, max_score = 0, -1
    # Iterate through every possible split point in the script window
    for i in range(len(segment) + 1):
        # Calculate similarity for the left side (current segment)
        s_left = SequenceMatcher(None, t_curr, segment[max(0, i-len(t_curr)):i].strip()).ratio() if t_curr else 1.0
        # Calculate similarity for the right side (start of next segment)
        s_right = SequenceMatcher(None, t_next, segment[i:i+len(t_next)].strip()).ratio() if t_next else 1.0
        
        # Combined score: higher is better
        score = (s_left + s_right) / 2
        if score > max_score:
            max_score, best_pos = score, i
    return best_pos, max_score

def run():
    # Check if necessary command line arguments are provided
    if len(sys.argv) < 3:
        print("USAGE: python srt_gen.py <audio_file> <script_file>")
        sys.exit(1)

    audio_in, script_arg = sys.argv[1], sys.argv[2]
    # Ensure script file has a .txt extension if not provided
    script_in = script_arg if os.path.exists(script_arg) else script_arg + ".txt"
    # Set up output directory based on script name and output SRT based on audio name
    output_dir = os.path.splitext(os.path.basename(script_in))[0]
    output_srt = os.path.splitext(os.path.basename(audio_in))[0] + ".srt"

    current_audio_workfile = audio_in
    temp_wav_file = None
    # Convert audio to 16kHz Mono WAV if it's not already in that format (required for Whisper)
    if not audio_in.lower().endswith(".wav"):
        fd, temp_wav_file = tempfile.mkstemp(suffix=".wav"); os.close(fd)
        subprocess.run(["ffmpeg", "-y", "-i", audio_in, "-ac", "1", "-ar", "16000", temp_wav_file], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        current_audio_workfile = temp_wav_file

    try:
        # Create a directory to store individual .txt files for each segment
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        
        # Load and fully clean the master script to use as a 'ruler'
        with open(script_in, "r", encoding="utf-8") as f:
            master_script = clean_text_fully(f.read())

        # Use FFmpeg to detect periods of silence to divide the audio into segments
        log_cmd = ["ffmpeg", "-i", current_audio_workfile, "-af", "silencedetect=noise=-30dB:d=0.5", "-f", "null", "-"]
        result = subprocess.run(log_cmd, stderr=subprocess.PIPE, text=True, errors="replace")
        s_starts = re.findall(r"silence_start: ([\d\.]+)", result.stderr)
        s_ends = re.findall(r"silence_end: ([\d\.]+)", result.stderr)

        # Map out the segments based on silence detections
        segments = []
        for j in range(len(s_ends)):
            start = float(s_ends[j])
            end = float(s_starts[j+1]) if j+1 < len(s_starts) else start + 5.0
            if end - start > 0.1: 
                segments.append({"start": start, "end": end, "dur": end - start})

        total_segs = len(segments)
        last_cut_pos = 0 # Tracks our current progress through the master script string
        success_history = []
        ai_model = None
        skipped_since_last_success = False

        def get_trans(s, d):
            """
            Internal helper to transcribe a specific audio slice using Whisper.
            """
            nonlocal ai_model
            if ai_model is None:
                # Load the high-performance 'turbo' model
                ai_model = whisper.load_model("turbo")
            
            # Extract temporary audio clip for transcription
            fd, tmp = tempfile.mkstemp(suffix=".wav"); os.close(fd)
            subprocess.run(["ffmpeg", "-y", "-ss", str(s), "-t", str(d), "-i", current_audio_workfile, 
                            "-ar", "16000", "-ac", "1", tmp], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Transcribe and immediately clean the text (remove AI-generated punctuation)
            txt = ai_model.transcribe(tmp)["text"].strip()
            os.remove(tmp)
            return clean_text_fully(txt)

        i = 0
        # Main Loop: Processing audio segments one by one
        while i < len(segments):
            seg = segments[i]
            # Transcribe current segment and peak at the next one for better split alignment
            t_curr = get_trans(seg['start'], seg['dur'])
            t_next_tentative = get_trans(segments[i+1]['start'], segments[i+1]['dur']) if i+1 < len(segments) else ""
            
            # Search for the heard text within a 1000-character window of the master script
            search_window = master_script[last_cut_pos : last_cut_pos + 1000]
            rel_split, score = get_refined_split_pos(t_curr, t_next_tentative, search_window, seg['dur'])

            # Only proceed if the similarity score exceeds our threshold
            if score >= SCORE_THRESHOLD:
                if success_history:
                    # 'Repair' the previous segment's end point based on where the current segment starts
                    prev = success_history[-1]
                    prev_window = master_script[prev['start_pos'] : prev['start_pos'] + 1000]
                    refined_pos, _ = get_refined_split_pos(prev['trans'], t_curr, prev_window, prev['dur'])
                    true_prev_text = prev_window[:refined_pos].strip()
                    
                    # Update the previous segment's file with cleaned/fixed text
                    with open(prev['path'], "w", encoding="utf-8") as f: f.write(true_prev_text)
                    last_cut_pos = prev['start_pos'] + refined_pos
                    
                    if skipped_since_last_success:
                        print(f"[{prev['index']:04d}/{total_segs:04d}] (Corrected) {true_prev_text[:70]}")

                # Save the current segment's script text to a uniquely named file (milliseconds timestamp)
                ts_ms = int(seg['start'] * 1000)
                txt_path = os.path.join(output_dir, f"{ts_ms:09d}.txt")
                current_window = master_script[last_cut_pos : last_cut_pos + 1000]
                final_rel_split, _ = get_refined_split_pos(t_curr, t_next_tentative, current_window, seg['dur'])
                final_text = current_window[:final_rel_split].strip()
                
                with open(txt_path, "w", encoding="utf-8") as f: f.write(final_text)

                # Store metadata for final SRT generation
                success_history.append({
                    'path': txt_path, 'start_pos': last_cut_pos, 'trans': t_curr, 
                    'dur': seg['dur'], 'start_time': seg['start'], 'end_time': seg['end'],
                    'index': i + 1
                })
                
                last_cut_pos += final_rel_split
                print(f"[{i+1:04d}/{total_segs:04d}] (Score:{score:.2f}) {final_text[:70]}")
                skipped_since_last_success = False
                i += 1
            else:
                # If score is too low, skip this segment and try to bridge the gap later
                print(f"[{i+1:04d}/{total_segs:04d}] (Score:{score:.2f}) SKIP")
                skipped_since_last_success = True
                i += 1 

        # Final Phase: Compile all success history into a valid SRT file
        srt_final = []
        for idx, h in enumerate(success_history):
            with open(h['path'], "r", encoding="utf-8") as f:
                txt = f.read()
            srt_final.append(f"{idx+1}\n{format_srt_time(h['start_time'])} --> {format_srt_time(h['end_time'])}\n{txt}\n")
        
        with open(output_srt, "w", encoding="utf-8") as f: f.write("\n".join(srt_final))
        print(f"Completed! SRT saved as: {output_srt}")

    finally:
        # Cleanup temporary audio files
        if temp_wav_file and os.path.exists(temp_wav_file): os.remove(temp_wav_file)

if __name__ == "__main__":
    run()

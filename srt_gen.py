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

# Minimum confidence score (0.0 to 1.0) to sync script and audio.
# High scores prevent the pointer from drifting during silence or filler words.
SCORE_THRESHOLD = 0.5

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def format_srt_time(seconds):
    """
    Converts float seconds into standard SRT timestamp format: HH:MM:SS,mmm.
    Ensures precise subtitle timing aligned with audio segments.
    """
    td = timedelta(seconds=seconds)
    total_sec = int(td.total_seconds())
    msec = int(td.microseconds / 1000)
    return f"{total_sec//3600:02}:{(total_sec%3600)//60:02}:{total_sec%60:02},{msec:03}"

def get_refined_split_pos(curr_heard, next_heard, script_segment, base_limit, extended_limit, long_wav_sec, dur):
    """
    Calculates the mathematically optimal split index in the script segment.
    Uses 'phonetic anchors' (tail of current audio and head of next audio) 
    to find the exact character position where the current segment ends.
    """
    # Extract the last 2 words of current transcription and first 2 of the next.
    words_next, words_curr = next_heard.split()[:2], curr_heard.split()[-2:]
    t_next, t_curr = " ".join(words_next), " ".join(words_curr)
    
    # Increase search limit for longer segments to handle complex sentences.
    limit = extended_limit if dur >= long_wav_sec else base_limit
    segment = script_segment[:limit]
    
    best_pos, max_score = 0, -1
    # Iterate through characters to find the point where phonetic similarity is maximized.
    for i in range(len(segment) + 1):
        s_left = SequenceMatcher(None, t_curr, segment[max(0, i-len(t_curr)):i].strip()).ratio() if t_curr else 0
        s_right = SequenceMatcher(None, t_next, segment[i:i+len(t_next)].strip()).ratio() if t_next else 0
        score = (s_left + s_right) / 2
        if score > max_score:
            max_score, best_pos = score, i
    return best_pos, max_score

# ==============================================================================
# MAIN ENGINE
# ==============================================================================

def run():
    # Ensure command line arguments are provided correctly.
    if len(sys.argv) < 3:
        print("USAGE: python srt_gen.py <audio_file> <script_file>")
        sys.exit(1)

    audio_in, script_arg = sys.argv[1], sys.argv[2]
    script_in = script_arg if os.path.exists(script_arg) else script_arg + ".txt"
    
    # Define output paths and working directory based on input names.
    audio_name = os.path.splitext(os.path.basename(audio_in))[0]
    output_dir = os.path.splitext(os.path.basename(script_in))[0]
    output_srt = audio_name + ".srt"

    # Pre-process audio: Convert to 16kHz Mono WAV (optimized for Whisper AI).
    current_audio_workfile = audio_in
    temp_wav_file = None
    if not audio_in.lower().endswith(".wav"):
        fd, temp_wav_file = tempfile.mkstemp(suffix=".wav"); os.close(fd)
        subprocess.run(["ffmpeg", "-y", "-i", audio_in, "-ac", "1", "-ar", "16000", temp_wav_file], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        current_audio_workfile = temp_wav_file

    try:
        # Load script and ensure it exists; normalize whitespace.
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        with open(script_in, "r", encoding="utf-8") as f:
            master_script = " ".join(f.read().split())

        # STEP 1: Voice Activity Detection (VAD) using FFmpeg silencedetect.
        # This identifies exactly where people are speaking vs silence.
        print(f"STEP 1: Mapping speech intervals...")
        log_cmd = ["ffmpeg", "-i", current_audio_workfile, "-af", "silencedetect=noise=-30dB:d=0.5", "-f", "null", "-"]
        result = subprocess.run(log_cmd, stderr=subprocess.PIPE, text=True, errors="replace")
        silence_starts = re.findall(r"silence_start: ([\d\.]+)", result.stderr)
        silence_ends = re.findall(r"silence_end: ([\d\.]+)", result.stderr)

        # Generate segments from the detected non-silent periods.
        segments = []
        for i in range(len(silence_ends)):
            start = float(silence_ends[i])
            end = float(silence_starts[i+1]) if i + 1 < len(silence_starts) else start + 5.0
            if end - start > 0.1: segments.append({"start": start, "end": end, "dur": end - start})

        total_count, ai_model, remaining_script, srt_data = len(segments), None, master_script, []
        
        # IMPORTANT: Track the index of the last segment that was successfully matched.
        # This allows us to jump back and update previous files when we recover from a SKIP.
        last_success_idx = -1

        for i, seg in enumerate(segments):
            curr_num = i + 1
            timestamp_ms = int(seg['start'] * 1000)
            txt_file_path = os.path.join(output_dir, f"{timestamp_ms:09d}.txt")
            final_segment_text, match_score = "", 0.0 

            # Load existing progress from disk if available to support resume.
            if os.path.exists(txt_file_path):
                with open(txt_file_path, "r", encoding="utf-8") as f: final_segment_text = f.read()
                if final_segment_text:
                    idx = remaining_script.find(final_segment_text)
                    if idx != -1: 
                        remaining_script = remaining_script[idx + len(final_segment_text):].lstrip()
                        last_success_idx = i
            else:
                # Load AI model lazily when needed for the first time.
                if ai_model is None:
                    ai_model = whisper.load_model("turbo")

                # Transcription helper: Cuts audio segment and feeds it to Whisper.
                def get_trans(s, d):
                    fd, tmp = tempfile.mkstemp(suffix=".wav"); os.close(fd)
                    subprocess.run(["ffmpeg", "-y", "-ss", str(s), "-t", str(d), "-i", current_audio_workfile, 
                                    "-ar", "16000", "-ac", "1", tmp], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    txt = ai_model.transcribe(tmp)["text"].strip(); os.remove(tmp); return txt

                # Transcribe current and next segment to verify boundary synchronization.
                t_curr = get_trans(seg['start'], seg['dur'])
                t_next = get_trans(segments[i+1]['start'], segments[i+1]['dur']) if curr_num < total_count else ""
                
                # Perform the character-level alignment against the master script.
                split_idx, match_score = get_refined_split_pos(t_curr, t_next, remaining_script, 45, 400, 5.5, seg['dur']) \
                                          if curr_num < total_count else (len(remaining_script), 1.0)

                if match_score >= SCORE_THRESHOLD:
                    # Sync point reached. Capture the text chunk from the current script pointer.
                    raw_block = remaining_script[:split_idx].strip()
                    
                    # Identify the 'anchor' (start of current speech) to isolate filler text.
                    anchor = " ".join(t_curr.split()[:2])
                    anchor_pos = raw_block.find(anchor)

                    # --- RETROACTIVE MERGING LOGIC ---
                    # If we just recovered from SKIPs, anchor_pos will be > 0.
                    # The text before anchor_pos is the 'unsettled' script that was missed during SKIP.
                    if anchor_pos > 0 and last_success_idx != -1:
                        bridge_trash = raw_block[:anchor_pos].strip()
                        current_valid = raw_block[anchor_pos:].strip()

                        if bridge_trash:
                            # 1. Update memory: Append trash to the end of the last confirmed SRT block.
                            prev_parts = srt_data[last_success_idx].split('\n')
                            prev_parts[2] = (prev_parts[2] + " " + bridge_trash).strip()
                            srt_data[last_success_idx] = "\n".join(prev_parts)

                            # 2. Update disk: Physically overwrite the previous .txt file with the merged text.
                            prev_ts = int(segments[last_success_idx]['start'] * 1000)
                            prev_path = os.path.join(output_dir, f"{prev_ts:09d}.txt")
                            with open(prev_path, "w", encoding="utf-8") as f_prev:
                                f_prev.write(prev_parts[2])
                        
                        final_segment_text = current_valid
                    else:
                        final_segment_text = raw_block

                    # Advance script pointer using absolute coordinates and clean leading spaces.
                    remaining_script = remaining_script[split_idx:].lstrip()
                    last_success_idx = i
                else:
                    # If score is low (filler audio), hold the pointer and skip script progression.
                    final_segment_text = ""

                # Cache the processed segment to disk.
                with open(txt_file_path, "w", encoding="utf-8") as f: f.write(final_segment_text)

            # Log current status with score and text preview.
            preview = (final_segment_text[:37] + "...") if len(final_segment_text) > 40 else final_segment_text
            log_msg = f"[{curr_num:04d}/{total_count:04d}] (Score:{match_score:.2f}) "
            print(f"{log_msg}{preview}" if final_segment_text else f"{log_msg}SKIP (Pointer Held)")

            # Store the (potentially later-modified) SRT block in memory.
            srt_data.append(f"{curr_num}\n{format_srt_time(seg['start'])} --> {format_srt_time(seg['end'])}\n{final_segment_text}\n")

        # Final assembly of the master SRT file.
        with open(output_srt, "w", encoding="utf-8") as f: f.write("\n".join(srt_data))
        print(f"\nFINISH: {output_srt}")
    finally:
        # Cleanup temporary audio files.
        if temp_wav_file and os.path.exists(temp_wav_file): os.remove(temp_wav_file)

if __name__ == "__main__":
    run()

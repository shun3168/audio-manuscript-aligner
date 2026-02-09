import os, subprocess, re, whisper, sys, tempfile, glob
from difflib import SequenceMatcher
from datetime import timedelta

SCORE_THRESHOLD = 0.3
RECOVERY_THRESHOLD = 0.9

total_files = 0

def format_srt_time(seconds):
    td = timedelta(seconds=seconds)
    total_sec = int(td.total_seconds())
    msec = int(td.microseconds / 1000)
    return f"{total_sec//3600:02}:{(total_sec%3600)//60:02}:{total_sec%60:02},{msec:03}"

def clean_text_fully(text):
    text = text.lower()
    text = re.sub(r'[「」『』、。！？!?,.，．…：；:;（）【】［］\(\)\[\]★◆▲●○◎♪■□#&%ー\-\'\"‘’“”]', ' ', text)
    text = re.sub(r'[\r\n\t　]+', ' ', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

def build_srt_from_txt_folder(output_dir, output_srt):
    txt_files = sorted(glob.glob(os.path.join(output_dir, "*.txt")))
    srt_final = []
    idx = 1
    for filepath in txt_files:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            continue
        time_line = os.path.splitext(os.path.basename(filepath))[0].replace("_", " --> ").replace(".", ",")
        srt_final.append(f"{idx}\n{time_line}\n{content}")
        idx += 1

    if srt_final:
        with open(output_srt, "w", encoding="utf-8") as f:
            f.write("\n\n".join(srt_final) + "\n")

def get_weighted_limit(dur, search_window):
    budget = dur * 30
    accumulated_cost = 0
    char_count = 0

    for char in search_window:
        if '\u4e00' <= char <= '\u9faf':
            cost = 5
        elif '\u3040' <= char <= '\u30ff' or '\uac00' <= char <= '\ud7a3':
            cost = 2
        else:
            cost = 1

        accumulated_cost += cost
        char_count += 1

        if accumulated_cost >= budget:
            break

    return max(char_count, 20)

def get_refined_split_pos(anchor_curr, anchor_next, script_segment, dur, is_recovery=False):
    if is_recovery:
        word_positions = [m.start() for m in re.finditer(r'\S+', script_segment)]
        word_positions.append(len(script_segment))
        
        t_next = " ".join(anchor_next)
        n_next = len(anchor_next)

        for char_idx in word_positions:
            if not t_next:
                s_right = 1.0
            else:
                sample_right = " ".join(script_segment[char_idx:].split()[:n_next])
                s_right = SequenceMatcher(None, t_next, sample_right).ratio()

            if s_right >= RECOVERY_THRESHOLD:
                return char_idx, s_right

        return 0, 0.0

    
    limit = get_weighted_limit(dur, script_segment)
    search_limit = min(limit, len(script_segment))
    
    segment = script_segment[:search_limit]

    word_positions = [m.start() for m in re.finditer(r'\S+', segment)]
    word_positions.append(len(segment))

    t_curr = " ".join(anchor_curr)
    t_next = " ".join(anchor_next)
    n_curr, n_next = len(anchor_curr), len(anchor_next)

    best_pos, max_score = 0, -1.0

    for char_idx in word_positions:
        if not t_curr:
            s_left = 1.0
        else:
            sample_left = " ".join(segment[:char_idx].split()[-n_curr:])
            s_left = SequenceMatcher(None, t_curr, sample_left).ratio()

        if not t_next:
            s_right = 1.0
        else:
            sample_right = " ".join(segment[char_idx:].split()[:n_next])
            s_right = SequenceMatcher(None, t_next, sample_right).ratio()

        if is_recovery:
            if s_right >= RECOVERY_THRESHOLD:
                return char_idx, s_right
        else:
            score = (s_left + s_right) / 2
            if score > max_score:
                max_score, best_pos = score, char_idx

    return best_pos, max_score

def prepare_resources(audio_in, script_in):
    try:
        with open(script_in, "r", encoding="utf-8-sig") as f:
            master_script = clean_text_fully(f.read())
    except Exception:
        print("USAGE: python srt_gen.py <audio_file> <script_file>")
        sys.exit(1)

    temp_wav_file = None

    probe_cmd = [
        "ffprobe", "-v", "error", "-show_entries", "stream=channels,sample_rate:format=duration",
        "-of", "csv=p=0", audio_in
    ]
    
    try:
        probe_res = subprocess.run(probe_cmd, capture_output=True, text=True).stdout.strip()
        if not probe_res:
            raise ValueError("No audio stream found")

        parts = [p.strip() for p in probe_res.replace('\n', ',').split(",")]
        
        channels = parts[0]
        sample_rate = parts[1]
        total_duration = float(parts[-1]) 

        is_optimal = (channels == "1" and sample_rate == "16000")

        if not is_optimal:
            fd, temp_wav_file = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            subprocess.run([
                "ffmpeg", "-y", "-i", audio_in,
                "-ac", "1", "-ar", "16000", temp_wav_file
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            current_audio_workfile = temp_wav_file
        else:
            current_audio_workfile = audio_in

    except Exception:
        print("ERROR: Failed to probe or convert audio file.")
        sys.exit(1)

    log_cmd = [
        "ffmpeg", "-i", current_audio_workfile,
        "-af", "silencedetect=noise=-30dB:d=0.3",
        "-f", "null", "-"
    ]
    result = subprocess.run(log_cmd, capture_output=True, text=True, errors="replace")
    
    s_starts = re.findall(r"silence_start: ([\d\.]+)", result.stderr)
    s_ends = re.findall(r"silence_end: ([\d\.]+)", result.stderr)

    segments = []
    for j, start_str in enumerate(s_ends):
        start = float(start_str)
        
        if (j + 1) < len(s_starts):
            end = float(s_starts[j+1])
        else:
            end = total_duration

        segments.append({"start": start, "end": end})

    global total_files
    total_files = len(segments)

    return current_audio_workfile, master_script, segments, temp_wav_file

def get_trans(seg, current_audio, ai_model):
    s = seg['start']
    d = (seg['end'] - s) if seg['end'] is not None else None
    
    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    
    cmd = ["ffmpeg", "-y", "-ss", f"{s:.3f}"]
    if d is not None:
        cmd += ["-t", f"{d:.3f}"]
    cmd += ["-i", current_audio, "-c", "copy", tmp]
    
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    result = ai_model.transcribe(tmp)
    txt = result["text"].strip()
    
    os.remove(tmp)
    return txt

import glob

def process_segment_helper(i, segments, master_script, output_dir, 
                           last_boundary_pos, skip_count, last_success_info, anchor_curr, anchor_next):
    global total_files
    
    seg = segments[i]
    s_t, e_t = seg['start'], seg['end']

    if skip_count >= 1:
        rel_split, score = get_refined_split_pos(
            anchor_curr, anchor_next, master_script[last_boundary_pos:], (e_t - last_success_info['start_time']), 
            is_recovery=True
        )
        if score == 0.0:
            print(f"[{i}/{total_files}] Recovery Failed")
            return skip_count + 1, last_boundary_pos
        current_text = master_script[last_boundary_pos : last_boundary_pos + rel_split]
        current_filename = f"{segments[i-skip_count+1]['start']:.3f}_{e_t:.3f}.txt"
        with open(os.path.join(output_dir, current_filename), "w", encoding="utf-8") as f:
            f.write(current_text)
        skip_count = 0
        print(f"[{i}/{total_files}] (Score:{score:.2f}) {current_text}")

        last_success_info['start_pos'] = last_boundary_pos
        last_success_info['start_time'] = segments[i]['start']
        last_boundary_pos += rel_split

        return 0, last_boundary_pos

    else:
        rel_split, score = get_refined_split_pos(
            anchor_curr, anchor_next, master_script[last_boundary_pos:], (e_t - s_t)
        )

    if score < SCORE_THRESHOLD:
        timestamp = format_srt_time(s_t).replace(',', '.')
        print(f"[{i}/{total_files}] (Score:{score:.2f}) SKIP {timestamp}")
        return skip_count + 1, last_boundary_pos

    current_text = master_script[last_boundary_pos : last_boundary_pos + rel_split]
    current_filename = f"{s_t:.3f}_{e_t:.3f}.txt"
    with open(os.path.join(output_dir, current_filename), "w", encoding="utf-8") as f:
        f.write(current_text)
    print(f"[{i}/{total_files}] (Score:{score:.2f}) {current_text}")

    last_success_info['start_pos'] = last_boundary_pos
    last_success_info['start_time'] = segments[i]['start']
    last_boundary_pos += rel_split

    return 0, last_boundary_pos

def run():
    print("Initializing...(1/3)")

    if len(sys.argv) < 3:
        print("USAGE: python srt_gen.py <audio_file> <script_file>")
        sys.exit(1)

    audio_in = sys.argv[1]
    script_in = sys.argv[2]

    script_base = os.path.splitext(os.path.basename(script_in))[0]
    audio_base = os.path.splitext(os.path.basename(audio_in))[0]

    output_dir = script_base
    output_srt = audio_base + ".srt"

    current_audio, master_script, segments, temp_wav = prepare_resources(audio_in, script_in)

    print("Initializing...(2/3)")

    if os.path.exists(output_dir):
        existing_files = os.listdir(output_dir)
        if existing_files:
            ans = input(f"Directory '{output_dir}' is not empty. Clear ALL files inside? (y/n): ").lower()
            if ans == 'y':
                print("Clearing all files...")
                for f in existing_files:
                    f_path = os.path.join(output_dir, f)
                    if os.path.isfile(f_path):
                        os.remove(f_path)
            else:
                print("Continuing with existing files...")
    else:
        os.makedirs(output_dir)

    try:
        if not os.path.exists(output_dir): 
            os.makedirs(output_dir)

        ai_model = whisper.load_model("turbo")

        print("Initializing...(3/3)")

        total_segs = len(segments)
        digit_width = len(str(total_segs))
 
        skip_count = 0
        last_boundary_pos = 0
        last_success_info = {'start_pos': 0, 'start_time': 0.0}
        
        raw_next = get_trans(segments[0], current_audio, ai_model)

        i = 0
        while i < len(segments):
            seg = segments[i]
            s_t, e_t = seg['start'], seg['end']

            if i == total_segs - 1:
                final_text = master_script[last_boundary_pos:]
                save_path = os.path.join(output_dir, f"{s_t:.3f}_{e_t:.3f}.txt")
                with open(save_path, "w", encoding="utf-8") as f: f.write(final_text)
                print(f"[{i}/{total_files}] (LAST) {final_text}")
                break

            if len(clean_text_fully(raw_next)) == 0:
                raw_curr = raw_next
                raw_next = get_trans(segments[i+1], current_audio, ai_model)
                anchor_curr = clean_text_fully(raw_curr).split()[-3:]
                anchor_next = clean_text_fully(raw_next).split()[:3]

                timestamp = format_srt_time(s_t).replace(',', '.')
                print(f"[{i}/{total_files}] (Silence) SKIP {timestamp}")
                for j in range(i, i - skip_count, -1):
                    segments[j] = segments[j-1]
                i += 1  
                continue

            if skip_count == 0:
                raw_curr = raw_next
            raw_next = get_trans(segments[i+1], current_audio, ai_model)
            anchor_curr = clean_text_fully(raw_curr).split()[-3:]
            anchor_next = clean_text_fully(raw_next).split()[:3]

            skip_count, last_boundary_pos = process_segment_helper(
                i, segments, master_script, 
                output_dir, last_boundary_pos, skip_count, last_success_info, anchor_curr, 
                anchor_next
            )

            i += 1

        build_srt_from_txt_folder(output_dir, output_srt)

    finally:
        if temp_wav and os.path.exists(temp_wav):
            os.remove(temp_wav)

if __name__ == "__main__":
    run()

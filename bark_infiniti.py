import numpy as np
from bark import SAMPLE_RATE, generate_audio, preload_models
import os
import datetime
import soundfile as sf
import re
from collections import defaultdict, namedtuple

FileData = namedtuple("FileData", ["filename", "name", "desc"])



SUPPORTED_LANGS = [
    ("English", "en"),
    ("German", "de"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("Hindi", "hi"),
    ("Italian", "it"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
    ("Polish", "pl"),
    ("Portuguese", "pt"),
    ("Russian", "ru"),
    ("Turkish", "tr"),
    ("Chinese", "zh"),
]



def read_npz_files(directory):
    return [f for f in os.listdir(directory) if f.endswith(".npz")]

def extract_name_and_desc(filepath):
    with np.load(filepath) as data:
        name = data.get('name', '')
        desc = data.get('desc', '')
        return name, desc

def categorize_files(files, directory):
    categorized_files = defaultdict(list)
    lang_dict = {code: lang for lang, code in SUPPORTED_LANGS}
    
    for file in files:
        name, desc = extract_name_and_desc(os.path.join(directory, file))
        match = re.match(r"([a-z]{2}|\w+)_", file)
        if match:
            prefix = match.group(1)
            if prefix in lang_dict:
                categorized_files[lang_dict[prefix]].append(FileData(file, name, desc))
            else:
                categorized_files[prefix.capitalize()].append(FileData(file, name, desc))
        else:
            categorized_files["Other"].append(FileData(file, name, desc))

    return categorized_files

# this is a mess but whatever
def print_speakers_list(categorized_files):
    print("Available history prompts:")
    for category, files in categorized_files.items():
        sorted_files = sorted(files, key=lambda x: (re.search(r"_\w+(_\d+)?\.npz$", x.filename) and re.search(r"_\w+(_\d+)?\.npz$", x.filename).group()[:-4], x.filename))
        print(f"\n  {category}:")
        for file_data in sorted_files:
            name_display = f'  "{file_data.name}"' if file_data.name else ''
            desc_display = f'{file_data.desc}' if file_data.desc else ''
            print(f"    {file_data.filename[:-4]} {name_display} {desc_display}")

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
history_prompt_dir = os.path.join(CUR_PATH, "bark", "assets", "prompts")

npz_files = read_npz_files(history_prompt_dir)
categorized_files = categorize_files(npz_files, history_prompt_dir)
ALLOWED_PROMPTS = {file[:-4] for file in npz_files}



def estimate_spoken_time(text, wpm=150, time_limit=14):
    # Remove text within square brackets
    text_without_brackets = re.sub(r'\[.*?\]', '', text)
    
    words = text_without_brackets.split()
    word_count = len(words)
    time_in_seconds = (word_count / wpm) * 60
    
    if time_in_seconds > time_limit:
        return True, time_in_seconds
    else:
        return False, time_in_seconds


def save_npz_file(filepath, x_semantic_continued, coarse_prompt, fine_prompt, output_dir=None):
    np.savez(filepath, semantic_prompt=x_semantic_continued, coarse_prompt=coarse_prompt, fine_prompt=fine_prompt)
    print(f"speaker file for this clip saved to {filepath}")

def split_text(text, split_words=0, split_lines=0):
    if split_words > 0:
        words = text.split()
        chunks = [' '.join(words[i:i + split_words]) for i in range(0, len(words), split_words)]
    elif split_lines > 0:
        lines = [line for line in text.split('\n') if line.strip()]
        chunks = ['\n'.join(lines[i:i + split_lines]) for i in range(0, len(lines), split_lines)]
    else:
        chunks = [text]
    return chunks

def save_audio_to_file(filepath, audio_array, sample_rate=24000, format='WAV', subtype='PCM_16', output_dir=None):
    sf.write(filepath, audio_array, sample_rate, format=format, subtype=subtype)
    print(f"Saved audio to {filepath}")


#def gen_and_save_audio(text_prompt, history_prompt=None, text_temp=0.7, waveform_temp=0.7, filename="", output_dir="bark_samples", split_by_words=0, split_by_lines=0, stable_mode=False, confused_travolta_mode=False, iteration=1):
def generate_long_audio (
    text: str,
    history_prompt: str = 'announcer',
):
  
    text_temp = 0
    waveform_temp = 0.7
    stable_mode = False
    confused_travolta_mode = False
    filename = ""
    output_dir = "bark_samples"

    print("Loading Bark models...")
    
    preload_models()

    print("Models loaded.")

    split_by_words = 35
    split_by_lines = 0

    text_chunks = split_text(text, split_by_words, split_by_lines)
    print(f"Processing chunk num: {len(text_chunks)}")

    audio_arr_chunks = []

    # Should output each audio chunk to disk midway so you at least a partial output if a long process crashes.
    for i, chunk in enumerate(text_chunks):
        print(f"Processing chunk {i + 1}/{len(text_chunks)}: {chunk}")
        longer_than_14_seconds, estimated_time = estimate_spoken_time(chunk)
        print(f"Current text chunk ballpark estimate: {estimated_time:.2f} seconds.")
        if longer_than_14_seconds:
            print(f"Text Prompt could be too long, might want to try a shorter one or try splitting tighter.")

        audio_array, x = generate_audio(text=chunk, history_prompt=history_prompt,text_temp=0)
        audio_arr_chunks.append(audio_array)
        
    concatenated_audio_arr = np.concatenate(audio_arr_chunks)
    return concatenated_audio_arr
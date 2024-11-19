from text import text_to_sequence, _clean_text
import commons
from torch import no_grad, LongTensor
import docx
import re

def split_text_by_length_or_character(text):
    lines = text.split('\n')
    processed_lines = []
    current_line = ""

    # for line in lines:
    #     if current_line:
    #         current_line += line
    #     else:
    #         current_line = line
        
    #     while len(current_line) >= 40:
    #         processed_lines.append(current_line[:40])
    #         current_line = current_line[40:]
    # return processed_lines
    
    if current_line:  # Append any remaining text as the last line
        processed_lines.append(current_line)
    pattern = r'(?<=[。！？ 、])|\n'

    sentences = re.split(pattern, text)
    sentences = [sentence for sentence in sentences if sentence ]
    return sentences


def get_text(text, hps, is_symbol):
    text_norm = text_to_sequence(text, hps.symbols, [] if is_symbol else hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm

def remove_empty_lines(text):
    lines = text.splitlines()
    cleaned_lines = [line for line in lines if line.strip()]
    cleaned_text = '\n'.join(cleaned_lines)
    return cleaned_text

def read_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

def read_docx_file(file_path):
    doc = docx.Document(file_path)
    content = []
    for paragraph in doc.paragraphs:
        content.append(paragraph.text)
    return '\n'.join(content)

# Gradio Interface
def file_or_text_reader(text, file):
    if file:
        # If the input is a file, determine its format and read its content
        if file.name.endswith('.txt'):
            content = read_txt_file(file.name)
        elif file.name.endswith('.docx'):
            content = read_docx_file(file.name)
        else:
            content = "Unsupported file format. Please upload a .txt or .docx file."
    if text:
        # If the input is text, return it directly
        content = text
    return content

def format_srt_entry(index, start_time, end_time, text):
    start = format_time(start_time)
    end = format_time(end_time)
    return f"{index}\n{start} --> {end}\n{text}\n"

def format_time(seconds):
    milliseconds = int((seconds % 1) * 1000)
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def save_srt_file(filename, content):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("\n".join(content))
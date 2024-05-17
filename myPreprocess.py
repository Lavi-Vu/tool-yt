import os
import argparse
import json
import sys


def write_transcripts_to_txt(source_wav_dir, speaker_name, transcript, output_file):
    # Create the full path to the directory containing the transcripts
    wav_file = "/parallel100/wav24kHz16bit/"
    transcript_path = source_wav_dir + speaker_name + transcript
    # Check if the directory exists
    if not os.path.exists(transcript_path):
        print(f"Directory '{transcript_path}' does not exist.")
        return
    
    # Initialize an empty string to store all transcripts
    all_transcripts = ""
    
    # Loop through all files in the transcript directory
    
    with open(output_file, 'a') as outfile:
        with open(transcript_path, 'r') as file:
            transcript_content = file.readlines()
            for content in transcript_content:
                content = source_wav_dir + speaker_name + wav_file + content.strip()
                speaker_id = speaker_name.replace("voice","")
                content = content.replace(":", ".wav|" + str(speaker_id) + "|[JA]") + '[JA]\n'
                file_name, line = content.split("|", 1)
                print(file_name)
            # Append transcript content to the string
                if os.path.exists(file_name):
                    outfile.write(content)           

    # # Write all transcripts to a text file
    # print(f"All transcripts from '{transcript_path}' have been written to '{output_file}'.")

if __name__ == "__main__":
    
    new_annos = []
    output_file = "/home/lavi/Documents/myprj/VITS-fast-fine-tuning/short_character_anno.txt"
    source_wav_dir = "/home/lavi/Documents/myprj/jvs_ver1/"
    speaker_names = list(os.walk(source_wav_dir))[0][1]
    transcript = "/parallel100/transcripts_utf8.txt"
    new_annos = []
    speakers = []
    
    for speaker_name in speaker_names:
        write_transcripts_to_txt(source_wav_dir, speaker_name, transcript, output_file)
        


            
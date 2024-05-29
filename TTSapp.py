import ffmpeg
import os
import pysrt
from datetime import timedelta

import ffmpeg
import os
import pysrt
from datetime import datetime, timedelta

def add_audio_and_subtitles_to_video(input_video_path, subtitle_file_path, audio_file_path, output_video_path):
    # Load subtitles
    subs = pysrt.open(subtitle_file_path)
    last_sub_end_time = subs[-1].end.to_time() if subs else timedelta(seconds=0)

    # Convert last_sub_end_time to timedelta for total_seconds calculation
    last_sub_end_time = timedelta(hours=last_sub_end_time.hour,
                                  minutes=last_sub_end_time.minute,
                                  seconds=last_sub_end_time.second,
                                  microseconds=last_sub_end_time.microsecond)
    
    # Get video duration
    probe = ffmpeg.probe(input_video_path)
    video_duration = float(probe['format']['duration'])

    # Calculate the total duration needed in seconds
    subtitle_end_time_sec = last_sub_end_time.total_seconds()

    # Construct the ffmpeg command
    input_video = ffmpeg.input(input_video_path, stream_loop=-1)
    input_audio = ffmpeg.input(audio_file_path)

    if os.path.exists(output_video_path):
        os.remove(output_video_path)

    # Apply subtitles
    video_with_subtitles = ffmpeg.filter_(input_video, 'subtitles', subtitle_file_path)
    
    # Trim the video to the end time of the subtitles
    trimmed_video = ffmpeg.trim(video_with_subtitles, duration=subtitle_end_time_sec)
    trimmed_video = ffmpeg.setpts(trimmed_video, 'PTS-STARTPTS')
    
    # Combine video with new audio, trimming audio to match the video length
    output = ffmpeg.output(trimmed_video, input_audio, output_video_path, 
                           vcodec='h264_nvenc', acodec='aac', 
                           video_bitrate='5M', strict='experimental',
                           shortest=None)
    
    ffmpeg.run(output)
    return output_video_path
# Example usage:
input_video_path = '/home/lavi/Downloads/1.mp4'
subtitle_file_path = 'create_video_demo/output.srt'
audio_file_path = 'create_video_demo/audio.wav'
output_video_path = 'create_video_demo/output.mp4'

add_audio_and_subtitles_to_video(input_video_path, subtitle_file_path, audio_file_path, output_video_path)

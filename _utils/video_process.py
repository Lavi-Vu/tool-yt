import pysrt
from datetime import timedelta

import ffmpeg
import os
def add_audio_and_subtitles_to_video(input_video_path, subtitle_file_path, audio_file_path, output_video_path):
    subs = pysrt.open(subtitle_file_path)
    last_sub_end_time = subs[-1].end.to_time() if subs else timedelta(seconds=0)

    # Convert last_sub_end_time to timedelta for total_seconds calculation
    last_sub_end_time = timedelta(hours=last_sub_end_time.hour,
                                  minutes=last_sub_end_time.minute,
                                  seconds=last_sub_end_time.second,
                                  microseconds=last_sub_end_time.microsecond)

    # Calculate the total duration needed in seconds
    subtitle_end_time_sec = last_sub_end_time.total_seconds()

    # Construct the ffmpeg command
    input_video = ffmpeg.input(input_video_path, stream_loop=-1)
    input_audio = ffmpeg.input(audio_file_path)

    if os.path.exists(output_video_path):
        os.remove(output_video_path)

    # Apply subtitles with black background
    video_with_subtitles = input_video.filter('subtitles', subtitle_file_path, force_style='Fontname=Roboto,OutlineColour=&H40000000,BorderStyle=3,FontSize=22')


    # Trim the video to the end time of the subtitles
    trimmed_video = ffmpeg.trim(video_with_subtitles, duration=subtitle_end_time_sec)
    trimmed_video = ffmpeg.setpts(trimmed_video, 'PTS-STARTPTS')

    # Combine video with new audio, trimming audio to match the video length
    output = ffmpeg.output(trimmed_video, input_audio, output_video_path,
                           vcodec='h264_nvenc', acodec='aac',
                           video_bitrate='5M', strict='experimental')

    ffmpeg.run(output)
    return output_video_path

def process(video):
    input_video_path = video.name
    subtitle_file_path = 'create_video_demo/output.srt'
    audio_file_path = 'create_video_demo/audio.wav'
    output_video_path = "create_video_demo/output_gradio_for_demo.mp4"

    # Add audio and subtitles to the video
    output_path = add_audio_and_subtitles_to_video(input_video_path, subtitle_file_path, audio_file_path, output_video_path)
    os.remove(input_video_path)
    return output_path

import pysrt
from datetime import timedelta
import shutil
import ffmpeg
import os
# from text_process
def add_audio_and_subtitles_to_video(input_video_path, subtitle_file_path, audio_file_path, output_video_path):
    subs = pysrt.open(subtitle_file_path)
    
    if (subtitle_file_path != 'create_video_demo/output.srt'):
        shutil.copyfile(subtitle_file_path, 'create_video_demo/output.srt')
    subtitle_file_path = 'create_video_demo/output.srt'
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
    video_with_subtitles = input_video.filter('subtitles', subtitle_file_path, force_style='Name=Default,Fontname=KOTRA_BOLD,Fontsize=20,PrimaryColour=&HFFFFFF&,SecondaryColour=&h0000FF,BackColour=&H0,BorderStyle=3,OutlineColour=&H80000000,Shadow=0')


    # Trim the video to the end time of the subtitles
    trimmed_video = ffmpeg.trim(video_with_subtitles, duration=subtitle_end_time_sec)
    trimmed_video = ffmpeg.setpts(trimmed_video, 'PTS-STARTPTS')

    # Combine video with new audio, trimming audio to match the video length
    output = ffmpeg.output(trimmed_video, input_audio, output_video_path,
                           vcodec='h264_nvenc', acodec='aac',
                           video_bitrate='4000k', strict='experimental')

    ffmpeg.run(output)
    return output_video_path

def process(video, checkbox, subtitle_file, audio_file):
    if checkbox == False:
        input_video_path = video.name
        subtitle_file_path = 'create_video_demo/output.srt'
        audio_file_path = 'create_video_demo/audio.wav'
        output_video_path = "create_video_demo/output_gradio_for_demo.mp4"
    else:
        input_video_path = video.name
        subtitle_file_path = subtitle_file.name
        audio_file_path = audio_file.name
        output_video_path = "create_video_demo/output_gradio_for_demo.mp4"

    # Add audio and subtitles to the video
    output_path = add_audio_and_subtitles_to_video(input_video_path, subtitle_file_path, audio_file_path, output_video_path)
    os.remove(input_video_path)
    return output_path

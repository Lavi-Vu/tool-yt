from faster_whisper import WhisperModel

model_size = "medium"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

def add_punctuation(text):
    punctuated_text = ""
    count = 0
    for char in text:
        count += 1
        punctuated_text += char
        if char == '。' or count % 40 ==0:  # Sentence-ending period in Japanese (full stop)
            punctuated_text += '\n'
            count = 0
    
    punctuated_text = punctuated_text.strip()
    if not punctuated_text.endswith('\n'):
        punctuated_text += '\n'
    return punctuated_text   
segments, info = model.transcribe("/home/lavi/Documents/myprj/VITS-fast-fine-tuning/create_video_demo/audio.wav", beam_size=5, language="ja")

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

text = str()

for segment in segments:
    text += segment.text
    # for word in segment.words:
    #     print("[%.2fs -> %.2fs] %s" % (word.start, word.end, word.word))
    
print(text)
print('=====================')

result2 = add_punctuation(text)
print(result2)
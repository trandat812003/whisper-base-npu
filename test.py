import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from RealtimeSTT import AudioToTextRecorder

def process_text(text):
    print(text)

if __name__ == '__main__':
    print("Wait until it says 'speak now'")
    recorder = AudioToTextRecorder(language="vi")

    while True:
        recorder.text(process_text)
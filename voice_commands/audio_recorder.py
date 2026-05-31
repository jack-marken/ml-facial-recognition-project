# from IPython import display
# import sounddevice as sd
# import soundfile as sf
# from playsound import playsound
# import numpy as np


import cv2
import pyaudio
import wave
import threading
import time
import subprocess
import os

class AudioRecorder:
    def __init__(self):
        self.open = True
        self.rate = 16000
        self.frames_per_buffer = 1024
        self.channels = 1
        self.format = pyaudio.paInt16
        self.audio_filename = "temp_audio.wav"
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=self.format,
                                      channels=self.channels,
                                      rate=self.rate,
                                      input=True,
                                      frames_per_buffer = self.frames_per_buffer)
        self.audio_frames = []
    
    # Audio starts being recorded
    def record(self):
        
        self.stream.start_stream()
        while(self.open == True):
            data = self.stream.read(self.frames_per_buffer) 
            self.audio_frames.append(data)
            if self.open==False:
                break

    # Finishes the audio recording therefore the thread too    
    def stop(self):
       
        if self.open==True:
            self.open = False
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()
               
            waveFile = wave.open(self.audio_filename, 'wb')
            waveFile.setnchannels(self.channels)
            waveFile.setsampwidth(self.audio.get_sample_size(self.format))
            waveFile.setframerate(self.rate)
            waveFile.writeframes(b''.join(self.audio_frames))
            waveFile.close()
        
        pass

SAMPLE_RATE = 16000  # Sample rate
duration = 1  # Duration of listening in seconds

stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    callback=audio_callback
)

try:
    with stream:
        while True:
            input("Press enter to record")
            print("Listening...")
            myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
            sd.wait()  # Wait until the recording is finished
            # display.display(display.Audio(myrecording, rate=16000))
            print("Playing...")
            sd.play(myrecording, 16000)
            sd.wait()
except KeyboardInterrupt:
    print("Stream stopped.")
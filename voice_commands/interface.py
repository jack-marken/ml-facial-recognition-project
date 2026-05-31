import time
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import threading
import pyaudio
import matplotlib
matplotlib.use('TkAgg') # Prevents matplotlib from interfering with cv2
import matplotlib.pyplot as plt
import cv2

# APIs from the local project files
# from audio_recorder import AudioRecorder

class UserInterface:
    def __init__(self):
        self.live_webcam_feed = cv2.VideoCapture(0)
        self.frame_rate = self.live_webcam_feed.get(cv2.CAP_PROP_FPS)
        # self.audio_thread = threading.Thread(target=self.record_command, daemon=True)

    # def record_command(self, current_video_frame: cv2.typing.MatLike):
        # Start audio in a separate thread

    def record_command(self):
        # print(fps)
        CHUNK = int(self.frame_rate)
        # CHUNK = capture.get(cv2.CAP_PROP_FPS)
        RATE = 16000
        LENGTH_SECONDS = 1

        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        # 1. Read raw bytes from the microphone
        print("Recording for 1 second...")
        frames = []

        # Calculate loops required for exactly 1 second
        for i in range(0, int(RATE / CHUNK * LENGTH_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(np.frombuffer(data, dtype=np.float32))

        npdata = np.hstack(frames)

        print("Finished recording.")

        # Pad the end of the numpy array so that it has a shape of (16000,)
        npdata = np.pad(npdata, (0, RATE - npdata.shape[0]), 'constant', constant_values=(0.0, 0.0))
        # print(npdata)
        # print(npdata.shape)

        # Clean up
        stream.stop_stream()
        stream.close()
        p.terminate()

        fig, axes = plt.subplots(2, figsize=(6, 4))
        timescale = np.arange(16000)
        axes[0].plot(timescale, npdata)
        axes[0].set_title('Waveform')
        axes[0].set_xlim([0, 16000])

        # # plot_spectrogram(spectrogram.numpy(), axes[1])
        # axes[1].set_title('Spectrogram')
        # # plt.suptitle(label.title())
        plt.show()

        return npdata

    # ====================================

        # SAMPLE_RATE = 16000
        # DURATION = 1.0
        # print("Recording...")
        # recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        # sd.wait()
        # # cv2.destroyWindow("Record Command") 
        # print("Playing back recording...")
        # sd.play(recording, SAMPLE_RATE)
        # sd.wait()
        # print("Done.")
        # print(recording)
        # self.audio_thread.stop()
    
    # def start_recording(self):
        # print("STARTED")
        # self.audio_thread.start()
        # print("ENDED?")
        # self.record_command(fps)

    def video_capture(self):
        # Initialise the video capture object to use the primary default webcam.

        print("Webcam initialised. Press 'q' in the video window to quit.")

        print("Webcam initialised. Press 'Enter' in the video window to register a new face.")

        # Begin an infinite loop to process the webcam feed frame by frame.
        while self.live_webcam_feed.isOpened():
            
            # Read the current frame from the webcam.
            successful_read, current_video_frame = self.live_webcam_feed.read()
            # print(current_video_frame.shape)
            
            # Check if the user presses the 'q' key to terminate the loop.
            keyboard_input = cv2.waitKey(1)
            if keyboard_input & 0xFF == ord('q'):
                break

            if keyboard_input & 0xFF == ord('r'):
                self.record_command()

                # frame_h, frame_w = current_video_frame.shape[:2]
                # frame_h, frame_w = 120, 400
                # color = (240, 250, 255)  # Off-white color in BGR format
                # thickness = -1       # -1 fills the rectangle completely

                # recording_window = np.zeros((frame_h, frame_w, 3), np.uint8)
                # recording_window[:] = color

                # message_text = "Recording command..."
                # font = cv2.FONT_HERSHEY_TRIPLEX
                # font_scale = 0.7
                # thickness = 2
                # (text_w, text_h), _ = cv2.getTextSize(message_text, font, font_scale, thickness)
                # text_x = (frame_w // 2) - (text_w // 2)
                # text_y = (frame_h + text_h) // 2
                # cv2.putText(recording_window, message_text, (text_x, text_y), font, font_scale, (26, 25, 24), thickness)
                # cv2.imshow("Record Command", recording_window)
                # cv2.waitKey(1000) # Wait 1 second while recording

                # # self.record_command(current_video_frame)
                # cv2.destroyWindow("Record Command")


            cv2.imshow("Face Detection Live Test", current_video_frame)


            # # cv2.imshow("Translucent box", output)

            # frame_w, frame_h = current_video_frame.shape[:2]
            # print(frame_w, frame_h)
            # cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=2)
            # self.record_command()

        # Release the webcam hardware and close all created graphical windows.
        self.live_webcam_feed.release()
        cv2.destroyAllWindows()

    def start(self):
        self.video_capture()

if __name__ == "__main__":
    app = UserInterface()
    app.video_capture()
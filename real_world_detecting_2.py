import os
import numpy as np
import librosa
import tensorflow as tf
import pyaudio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import queue
import threading
import time

# Load the saved trained model
model_path = r"C:\Users\Pc\Documents\Ml_Project\scream_1d_cnn_model.h5"  # Replace with your model path
model = tf.keras.models.load_model(model_path)

# Audio parameters
CHUNK = 32000  # 2 seconds at 16kHz
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000
DURATION = 2.0

# Queue for audio data
audio_queue = queue.Queue()

# Function to load and preprocess audio chunk
def process_audio_chunk(audio_data):
    audio = np.frombuffer(audio_data, dtype=np.float32)
    if len(audio) < CHUNK:
        audio = np.pad(audio, (0, CHUNK - len(audio)), mode='constant')
    return audio[:CHUNK]

# Prediction function
def predict_scream(audio):
    audio_input = audio[np.newaxis, ..., np.newaxis]  # Shape: (1, 32000, 1)
    prediction = model.predict(audio_input, verbose=0)[0][0]
    is_scream = prediction > 0.5
    label = "Scream" if is_scream else "Not a Scream"
    confidence = prediction if is_scream else 1 - prediction
    return label, confidence, is_scream

# Audio recording thread
def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    print("Recording started...")
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_queue.put(data)
    
    stream.stop_stream()
    stream.close()
    p.terminate()

# Real-time plotting and detection
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
t = np.linspace(0, DURATION, CHUNK)
line1, = ax1.plot(t, np.zeros(CHUNK), 'b')
hist_data = np.zeros(CHUNK)
bars = ax2.hist(hist_data, bins=50, color='blue', alpha=0.7, density=True)[1]
ax1.set_title("Waveform")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Amplitude")
ax2.set_title("Amplitude Distribution")
ax2.set_xlabel("Amplitude")
ax2.set_ylabel("Density")
ax1.grid(True)
ax2.grid(True)

def update_plot(frame):
    if not audio_queue.empty():
        audio_data = audio_queue.get()
        audio = process_audio_chunk(audio_data)
        
        # Predict scream
        label, confidence, is_scream = predict_scream(audio)
        
        # Update plot color based on scream detection
        color = 'red' if is_scream else 'blue'
        line1.set_color(color)
        line1.set_ydata(audio)
        
        # Update histogram
        ax2.clear()
        ax2.hist(audio, bins=50, color=color, alpha=0.7, density=True)
        ax2.set_title(f"Amplitude Distribution - {label}")
        ax2.set_xlabel("Amplitude")
        ax2.set_ylabel("Density")
        ax2.grid(True)
        
        # Adjust y-limits for waveform
        ax1.set_ylim(np.min(audio) - 0.1, np.max(audio) + 0.1)
        
        # Print prediction
        print(f"\n{'='*50}")
        print(f"Prediction: \033[1;{31 if is_scream else 34}m{label}\033[0m (Confidence: {confidence:.4f})")
        print(f"{'='*50}\n")
        
        # Trigger emergency message if scream detected
        if is_scream:
            print("\033[1;31mEMERGENCY: Scream Detected! Initiating alert...\033[0m")
            # Add your emergency action here (e.g., send SMS, email, etc.)
        
        # Audio Features
        print(f"{'-'*50}")
        print("Audio Features:")
        print(f"- Max Amplitude: {np.max(np.abs(audio)):.4f}")
        print(f"- Mean Amplitude: {np.mean(np.abs(audio)):.4f}")
        print(f"- Standard Deviation: {np.std(audio):.4f}")
        print(f"- Duration: {DURATION} seconds")
        print(f"- Sample Rate: {RATE} Hz")
        print(f"{'-'*50}")
    
    return line1,

# Start recording in a separate thread
recording_thread = threading.Thread(target=record_audio, daemon=True)
recording_thread.start()

# Animate the plot
ani = FuncAnimation(fig, update_plot, interval=100, blit=False)
plt.tight_layout()
plt.show()
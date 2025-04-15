import os
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, flash, redirect, url_for
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
import time

# Initialize Flask and SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'ScreamdetectionAnfas'
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max file size: 16MB
socketio = SocketIO(app)

MODEL_PATH = "scream_1d_cnn_model.h5"  
model = tf.keras.models.load_model(MODEL_PATH)



SR = 16000 # CONVERTING AUDIO INTO 16HZ
CHUNK_DURATION = 2.0
CHUNK_SIZE = int(SR * CHUNK_DURATION) # Calculates total number of samples per chunk (e.g., 16,000 * 2 = 32,000 samples).
ALLOWED_EXTENSIONS = {'wav', 'mp3'}
audio_buffer = []

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS #CHECKING THE EXTENTION OF THE AUDIO SAMPLES

def preprocess_audio(audio_data):
    if len(audio_data) < CHUNK_SIZE:
        audio_data = np.pad(audio_data, (0, CHUNK_SIZE - len(audio_data)), mode='constant') # Padding the audio data to ensure it has the same length as CHUNK_SIZE
    return audio_data[:CHUNK_SIZE]

def load_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SR, duration=CHUNK_DURATION, mono=True)
        return preprocess_audio(y)
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return None

def generate_plots(audio_data, label):
    try:
        sr, duration = 16000, 2.0
        t = np.linspace(0, duration, int(sr * duration))
        timestamp = int(time.time() * 1000)  # Unique timestamp for filenames

        # Waveform
        waveform_filename = f'waveform_{timestamp}.png'
        waveform_path = os.path.join('static', waveform_filename)
        plt.figure(figsize=(12, 4))
        plt.plot(t, audio_data, color='red')
        plt.title(f'Waveform - {label}')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.savefig(waveform_path)
        plt.close()
        print(f"Waveform saved at: {waveform_path}")

        # Histogram
        hist_filename = f'histogram_{timestamp}.png'
        hist_path = os.path.join('static', hist_filename)
        plt.figure(figsize=(6, 4))
        plt.hist(audio_data, bins=50, color='red', alpha=0.7, density=True)
        plt.title(f'Amplitude Distribution - {label}')
        plt.xlabel('Amplitude')
        plt.ylabel('Density')
        plt.grid(True)
        plt.savefig(hist_path)
        plt.close()
        print(f"Histogram saved at: {hist_path}")

        if os.path.exists(waveform_path) and os.path.exists(hist_path):
            return waveform_filename, hist_filename
        else:
            print("Files not found after saving!")
            return None, None
    except Exception as e:
        print(f"Error generating plots: {e}")
        return None, None

def analyze_audio(audio_data):
    audio = preprocess_audio(audio_data)
    audio_input = audio[np.newaxis, ..., np.newaxis]
    prediction = model.predict(audio_input, verbose=0)[0][0]
    is_scream = prediction > 0.5
    label = "Scream" if is_scream else "Not a Scream"
    confidence = prediction if is_scream else 1 - prediction

    features = {
        "max_amplitude": float(np.max(np.abs(audio))),
        "mean_amplitude": float(np.mean(np.abs(audio))),
        "std_dev": float(np.std(audio))
    }

    plots = None
    if is_scream:
        waveform_filename, hist_filename = generate_plots(audio, label)
        if waveform_filename and hist_filename:
            plots = {"waveform": waveform_filename, "histogram": hist_filename}
        else:
            print("Plot generation failed, skipping plots.")

    return label, confidence, features, plots

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            audio_data = load_audio(file_path)
            if audio_data is None:
                flash('Error processing audio file')
                os.remove(file_path)
                return redirect(request.url)

            label, confidence, features, plots = analyze_audio(audio_data)
            os.remove(file_path)

            return render_template('index.html', 
                                 uploaded=True, 
                                 label=label, 
                                 confidence=confidence, 
                                 features=features, 
                                 plots=plots)
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('status', {'message': 'Connected to server'})

@socketio.on('audio_data')
def handle_audio_data(data):
    global audio_buffer
    audio_chunk = np.frombuffer(data, dtype=np.float32)
    audio_buffer.extend(audio_chunk)

    if len(audio_buffer) >= CHUNK_SIZE:
        audio_data = np.array(audio_buffer[:CHUNK_SIZE])
        audio_buffer = audio_buffer[CHUNK_SIZE:]

        label, confidence, features, plots = analyze_audio(audio_data)
        response = {
            'label': label,
            'confidence': f"{confidence:.4f}",
            'features': features
        }
        if plots:
            response['plots'] = plots
        emit('prediction', response)

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
    global audio_buffer
    audio_buffer = []

if __name__ == '__main__':
    
    socketio.run(app, debug=True, host='127.0.0.1', port=5000)
"""
Flask Web Application for Speech Emotion Recognition
Upload audio files to detect emotions in real-time
"""

from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import librosa
import pickle
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for model and encoder
model = None
label_encoder = None

def load_models():
    """Load the trained model and label encoder"""
    global model, label_encoder
    
    try:
        model = load_model('models/best_model.h5')
        with open('models/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        print("Models loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

def extract_features(file_path):
    """
    Extract audio features from a file
    
    Args:
        file_path: Path to the audio file
    
    Returns:
        numpy array of concatenated features
    """
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast', 
                                         duration=2.5, sr=22050, offset=0.5)
        
        result = np.array([])
        
        # MFCC
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
        
        # Chroma
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma))
        
        # Mel spectrogram
        mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel))
        
        return result
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def predict_emotion(file_path):
    """
    Predict emotion from audio file
    
    Args:
        file_path: Path to the audio file
    
    Returns:
        Dictionary with emotion and confidence scores
    """
    if model is None or label_encoder is None:
        return None
    
    # Extract features
    features = extract_features(file_path)
    if features is None:
        return None
    
    # Reshape for prediction
    features = np.array([features])
    
    # Predict
    predictions = model.predict(features, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class] * 100
    
    # Get emotion name
    emotion = label_encoder.inverse_transform([predicted_class])[0]
    
    # Get all probabilities (excluding 'calm')
    allowed_emotions = ['angry', 'disgust', 'fearful', 'happy', 'sad', 'neutral', 'surprised']
    all_emotions = {}
    for idx, prob in enumerate(predictions[0]):
        emotion_name = label_encoder.inverse_transform([idx])[0]
        if emotion_name.lower() in allowed_emotions:
            all_emotions[emotion_name] = float(prob * 100)
    
    # If predicted emotion is 'calm', find next best emotion
    if emotion.lower() == 'calm':
        # Get the highest probability from allowed emotions
        best_emotion = max(all_emotions.items(), key=lambda x: x[1])
        emotion = best_emotion[0]
        confidence = best_emotion[1]
    
    return {
        'emotion': emotion,
        'confidence': float(confidence),
        'all_probabilities': all_emotions
    }

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle audio file upload and prediction"""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.endswith(('.wav', '.mp3', '.ogg', '.flac')):
        return jsonify({'error': 'Invalid file format. Please upload WAV, MP3, OGG, or FLAC files'}), 400
    
    try:
        # Save file
        filename = 'uploaded_audio.wav'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Predict emotion
        result = predict_emotion(filepath)
        
        # Clean up
        if os.path.exists(filepath):
            os.remove(filepath)
        
        if result is None:
            return jsonify({'error': 'Error processing audio file'}), 500
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'encoder_loaded': label_encoder is not None
    })

if __name__ == '__main__':
    print("Loading models...")
    if load_models():
        print("Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Error: Could not load models. Please train the model first using train_model.py")

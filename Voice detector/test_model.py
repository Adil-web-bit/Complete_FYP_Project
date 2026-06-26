"""
Quick test script to verify the trained model works
"""
import numpy as np
import librosa
import pickle
from tensorflow import keras

def extract_features(file_path, sample_rate=22050):
    """Extract audio features"""
    try:
        audio, sr = librosa.load(file_path, sr=sample_rate, duration=3)
        
        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        
        # Extract Chroma (using pre-computed STFT)
        stft = np.abs(librosa.stft(audio))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sr, tuning=0)
        chroma_mean = np.mean(chroma.T, axis=0)
        
        # Extract Mel spectrogram
        mel = librosa.feature.melspectrogram(y=audio, sr=sr)
        mel_mean = np.mean(mel.T, axis=0)
        
        # Combine features
        features = np.hstack([mfcc_mean, chroma_mean, mel_mean])
        return features
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def test_model(audio_file):
    """Test the model on a single audio file"""
    print(f"\n{'='*60}")
    print(f"Testing model with: {audio_file}")
    print(f"{'='*60}")
    
    # Load model
    print("\n1. Loading model...")
    model = keras.models.load_model('models/emotion_model.h5')
    
    # Load label encoder
    print("2. Loading label encoder...")
    with open('models/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Extract features
    print("3. Extracting features from audio...")
    features = extract_features(audio_file)
    
    if features is None:
        print("❌ Failed to extract features!")
        return
    
    print(f"   Feature shape: {features.shape}")
    
    # Make prediction
    print("4. Making prediction...")
    features = features.reshape(1, -1)
    prediction = model.predict(features, verbose=0)
    
    # Get predicted emotion
    predicted_class = np.argmax(prediction, axis=1)
    predicted_emotion = label_encoder.inverse_transform(predicted_class)[0]
    confidence = prediction[0][predicted_class[0]] * 100
    
    # Display results
    print(f"\n{'='*60}")
    print(f"📊 PREDICTION RESULTS")
    print(f"{'='*60}")
    print(f"🎭 Predicted Emotion: {predicted_emotion.upper()}")
    print(f"📈 Confidence: {confidence:.2f}%")
    print(f"\n📋 All Probabilities:")
    
    # Sort predictions by probability
    emotion_probs = list(zip(label_encoder.classes_, prediction[0]))
    emotion_probs.sort(key=lambda x: x[1], reverse=True)
    
    for emotion, prob in emotion_probs:
        bar_length = int(prob * 50)
        bar = '█' * bar_length + '░' * (50 - bar_length)
        print(f"   {emotion:12s} [{bar}] {prob*100:5.2f}%")
    
    print(f"{'='*60}\n")

if __name__ == "__main__":
    # Test with a sample file
    test_file = "Dataset/Actor_01/03-01-01-01-01-01-01.wav"
    test_model(test_file)
    
    # Test with a few more files
    import glob
    sample_files = glob.glob("Dataset/Actor_01/*.wav")[:3]
    
    print(f"\n{'='*60}")
    print(f"Testing with 3 sample audio files:")
    print(f"{'='*60}\n")
    
    for i, file in enumerate(sample_files, 1):
        print(f"\n--- Test {i}/3 ---")
        test_model(file)

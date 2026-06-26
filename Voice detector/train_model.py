"""
Speech Emotion Recognition Model Training Script
This script trains a deep learning model to detect emotions from audio files.
"""

import os
import numpy as np
import pandas as pd
import librosa
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, LSTM, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Emotion mapping based on RAVDESS dataset filename convention
EMOTIONS = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def extract_features(file_path, mfcc=True, chroma=True, mel=True):
    """
    Extract audio features from a file
    
    Args:
        file_path: Path to the audio file
        mfcc: Extract MFCC features
        chroma: Extract Chroma features
        mel: Extract Mel spectrogram features
    
    Returns:
        numpy array of concatenated features
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            audio, sample_rate = librosa.load(file_path, res_type='soxr_hq', duration=2.5, sr=22050, offset=0.5)
        
        result = np.array([])
        
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
        
        return result
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def load_dataset(dataset_path):
    """
    Load and extract features from all audio files in the dataset
    
    Args:
        dataset_path: Path to the dataset directory
    
    Returns:
        features: numpy array of features
        labels: numpy array of emotion labels
    """
    features = []
    labels = []
    
    # Get all actor directories
    actor_dirs = [d for d in os.listdir(dataset_path) if d.startswith('Actor_')]
    
    print("Loading dataset and extracting features...")
    
    for actor_dir in tqdm(actor_dirs, desc="Processing actors"):
        actor_path = os.path.join(dataset_path, actor_dir)
        
        if not os.path.isdir(actor_path):
            continue
        
        # Process each audio file
        for file in os.listdir(actor_path):
            if file.endswith('.wav'):
                file_path = os.path.join(actor_path, file)
                
                # Extract emotion from filename (3rd position in filename)
                emotion_code = file.split('-')[2]
                emotion = EMOTIONS.get(emotion_code, 'unknown')
                
                if emotion == 'unknown':
                    continue
                
                # Extract features
                feature = extract_features(file_path)
                
                if feature is not None:
                    features.append(feature)
                    labels.append(emotion)
    
    return np.array(features), np.array(labels)

def create_model(input_shape, num_classes):
    """
    Create a CNN-LSTM hybrid model for emotion classification
    
    Args:
        input_shape: Shape of input features
        num_classes: Number of emotion classes
    
    Returns:
        Compiled Keras model
    """
    model = Sequential()
    
    # Reshape for Conv1D
    model.add(Dense(256, input_shape=(input_shape,)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    """Main training function"""
    
    # Set dataset path
    dataset_path = 'Dataset'
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset directory '{dataset_path}' not found!")
        return
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Load and extract features
    print("Step 1: Loading dataset and extracting features...")
    X, y = load_dataset(dataset_path)
    
    print(f"\nDataset loaded successfully!")
    print(f"Total samples: {len(X)}")
    print(f"Feature shape: {X.shape}")
    print(f"Emotions distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for emotion, count in zip(unique, counts):
        print(f"  {emotion}: {count}")
    
    # Encode labels
    print("\nStep 2: Encoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    
    # Save label encoder
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print("Label encoder saved to models/label_encoder.pkl")
    
    # Split dataset
    print("\nStep 3: Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Create model
    print("\nStep 4: Creating model...")
    model = create_model(X_train.shape[1], y_categorical.shape[1])
    print(model.summary())
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
        'models/best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # Train model
    print("\nStep 5: Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
        verbose=1
    )
    
    # Evaluate model
    print("\nStep 6: Evaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Save final model
    model.save('models/emotion_model.h5')
    print("\nModel saved to models/emotion_model.h5")
    
    # Save training history
    with open('models/training_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    print("Training history saved to models/training_history.pkl")
    
    print("\n" + "="*50)
    print("Training completed successfully!")
    print("="*50)
    print("\nGenerated files:")
    print("  - models/emotion_model.h5 (trained model)")
    print("  - models/best_model.h5 (best model during training)")
    print("  - models/label_encoder.pkl (label encoder)")
    print("  - models/training_history.pkl (training history)")

if __name__ == '__main__':
    main()

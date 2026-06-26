# 🎭 Speech Emotion Recognition Project

A complete AI-powered system for detecting emotions from speech audio files using deep learning.

## 📋 Overview

This project uses a deep neural network to classify human emotions from audio recordings. It can detect 8 different emotions:
- 😐 Neutral
- 😌 Calm
- 😊 Happy
- 😢 Sad
- 😠 Angry
- 😨 Fearful
- 🤢 Disgust
- 😲 Surprised

## 🌟 Features

- **Deep Learning Model**: CNN-based architecture for accurate emotion classification
- **Audio Feature Extraction**: Uses MFCC, Chroma, and Mel spectrogram features
- **Web Interface**: Beautiful, user-friendly web application
- **Real-time Prediction**: Upload audio files and get instant results
- **Confidence Scores**: See probability distribution across all emotions
- **Multiple Format Support**: WAV, MP3, OGG, FLAC files

## 📁 Project Structure

```
Voice detector/
│
├── Dataset/                    # Training dataset (RAVDESS)
│   ├── Actor_01/
│   ├── Actor_02/
│   └── ...
│
├── models/                     # Saved models (created after training)
│   ├── emotion_model.h5
│   ├── best_model.h5
│   ├── label_encoder.pkl
│   └── training_history.pkl
│
├── templates/                  # HTML templates
│   └── index.html
│
├── uploads/                    # Temporary upload folder (auto-created)
│
├── train_model.py             # Model training script
├── app.py                     # Flask web application
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Navigate to Project Directory

```bash
cd "C:\Users\AT\OneDrive - University of Engineering and Technology Taxila\Desktop\Voice detector"
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: Installation may take 5-10 minutes as it includes TensorFlow and audio processing libraries.

## 🎓 Training the Model

Before running the web application, you need to train the model on your dataset.

### Run Training Script

```bash
python train_model.py
```

**Training Process:**
1. Loads audio files from the `Dataset` folder
2. Extracts audio features (MFCC, Chroma, Mel spectrogram)
3. Trains a deep neural network
4. Saves the trained model to `models/` folder

**Expected Training Time**: 30-60 minutes (depending on your hardware)

**Training Output:**
- `models/emotion_model.h5` - Final trained model
- `models/best_model.h5` - Best model checkpoint
- `models/label_encoder.pkl` - Label encoder for emotions
- `models/training_history.pkl` - Training metrics

## 🌐 Running the Web Application

After training the model, start the web server:

```bash
python app.py
```

Then open your web browser and navigate to:
```
http://localhost:5000
```

## 📱 Using the Web Interface

1. **Upload Audio**: Click or drag-and-drop an audio file
2. **Analyze**: Click "Analyze Emotion" button
3. **View Results**: See the detected emotion with confidence score
4. **Explore Probabilities**: Check the probability distribution of all emotions

## 🔧 Technical Details

### Model Architecture

- **Input**: 168-dimensional feature vector (40 MFCC + 12 Chroma + 128 Mel features)
- **Architecture**: 
  - Dense layers with batch normalization
  - Dropout for regularization
  - Softmax output layer for 8 emotions
- **Optimizer**: Adam
- **Loss Function**: Categorical Cross-entropy

### Feature Extraction

- **MFCC** (Mel-frequency cepstral coefficients): 40 coefficients
- **Chroma**: 12 features
- **Mel Spectrogram**: 128 features
- **Audio Processing**: 2.5 seconds duration, 22050 Hz sample rate

### Dataset

The model is trained on the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset:
- 24 professional actors
- 8 emotions
- ~1,440 audio files total

## 📊 Model Performance

After training, you should expect:
- **Training Accuracy**: 85-95%
- **Validation Accuracy**: 70-80%
- **Test Accuracy**: 65-75%

*Note: Actual performance may vary based on dataset quality and training parameters.*

## 🛠️ Troubleshooting

### Model Not Found Error
- Make sure you've run `train_model.py` before starting the web app
- Check that `models/best_model.h5` exists

### Audio Processing Error
- Ensure your audio file is in a supported format (WAV, MP3, OGG, FLAC)
- Try converting to WAV format if issues persist

### Installation Issues
- Update pip: `python -m pip install --upgrade pip`
- Install Visual C++ Build Tools (for Windows)
- Try installing packages individually if batch installation fails

## 📝 API Endpoints

### `GET /`
Returns the main web interface

### `POST /predict`
Accepts audio file and returns emotion prediction
- **Input**: Audio file (multipart/form-data)
- **Output**: JSON with emotion, confidence, and all probabilities

### `GET /health`
Health check endpoint
- **Output**: Server status and model loading status

## 🔮 Future Enhancements

- [ ] Real-time microphone input
- [ ] Support for multiple languages
- [ ] Emotion trend analysis over time
- [ ] Mobile app version
- [ ] Batch processing of multiple files
- [ ] Export results to CSV/PDF

## 📄 License

This project is for educational purposes. The RAVDESS dataset has its own license terms.

## 🤝 Contributing

Feel free to fork, improve, and submit pull requests!

## 📧 Contact

For questions or issues, please create an issue in the repository.

---

**Made with ❤️ using Python, TensorFlow, and Flask**

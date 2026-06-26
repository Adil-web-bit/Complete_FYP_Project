# Speech Emotion Recognition - Quick Start Guide

## 🚀 Getting Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train the Model
```bash
python train_model.py
```
*This will take 30-60 minutes. The script will:*
- Load all audio files from Dataset folder
- Extract features from audio
- Train a deep learning model
- Save the trained model to `models/` folder

### Step 3: Run the Web Application
```bash
python app.py
```
*Then open http://localhost:5000 in your browser*

---

## 📋 Optional: Check Your Setup

Before training, verify everything is set up correctly:
```bash
python setup_check.py
```

This will check:
- ✅ Python version
- ✅ Installed dependencies
- ✅ Dataset availability
- ✅ Trained models (after training)

---

## 🎯 Using the Web Interface

1. Open http://localhost:5000
2. Upload an audio file (WAV, MP3, OGG, or FLAC)
3. Click "Analyze Emotion"
4. View the detected emotion with confidence scores!

---

## 📁 Project Files

- `train_model.py` - Train the emotion detection model
- `app.py` - Flask web application
- `setup_check.py` - Verify your setup
- `requirements.txt` - Python dependencies
- `README.md` - Complete documentation

---

## ⚡ Troubleshooting

**"Models not found" error?**
- Run `python train_model.py` first to train the model

**Dependencies installation fails?**
- Update pip: `python -m pip install --upgrade pip`
- Install packages one by one if needed

**Audio processing errors?**
- Convert your audio to WAV format
- Ensure audio duration is at least 1 second

---

## 🎭 Supported Emotions

- 😐 Neutral
- 😌 Calm  
- 😊 Happy
- 😢 Sad
- 😠 Angry
- 😨 Fearful
- 🤢 Disgust
- 😲 Surprised

---

**Need help? Check README.md for detailed documentation!**

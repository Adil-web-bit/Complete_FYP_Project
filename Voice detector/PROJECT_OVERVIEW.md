# 🎭 Speech Emotion Recognition - Complete Project

## 📦 What You Have

A complete AI-powered emotion detection system that can:
- Detect 8 emotions from speech audio
- Train on your RAVDESS dataset
- Provide a beautiful web interface for predictions
- Show confidence scores and probability distributions

## 📂 Complete File Structure

```
Voice detector/
│
├── 📊 Dataset/                     # Your training data
│   ├── Actor_01/
│   ├── Actor_02/
│   └── ... (24 actors total)
│
├── 🤖 Python Scripts
│   ├── train_model.py              # Model training (30-60 min)
│   ├── app.py                      # Flask web application
│   └── setup_check.py              # Verify your setup
│
├── 🌐 Web Interface
│   └── templates/
│       └── index.html              # Beautiful UI
│
├── 📝 Documentation
│   ├── README.md                   # Complete guide
│   ├── QUICKSTART.md              # Quick start guide
│   └── PROJECT_OVERVIEW.md        # This file
│
├── ⚙️ Configuration
│   └── requirements.txt            # Python packages
│
└── 🚀 Batch Files (Windows)
    ├── install.bat                 # Install dependencies
    ├── train.bat                   # Train the model
    ├── run.bat                     # Start web app
    └── check.bat                   # Check setup

After training, you'll also have:
└── 💾 models/                      # Saved models
    ├── emotion_model.h5
    ├── best_model.h5
    ├── label_encoder.pkl
    └── training_history.pkl
```

## 🎯 How to Use This Project

### Option 1: Using Batch Files (Easiest for Windows)

1. **Install Dependencies**
   ```
   Double-click: install.bat
   ```

2. **Train the Model**
   ```
   Double-click: train.bat
   Wait 30-60 minutes
   ```

3. **Run Web Application**
   ```
   Double-click: run.bat
   Open: http://localhost:5000
   ```

### Option 2: Using Command Line

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Check Setup**
   ```bash
   python setup_check.py
   ```

3. **Train Model**
   ```bash
   python train_model.py
   ```

4. **Run Web App**
   ```bash
   python app.py
   ```

## 🔍 What Each Component Does

### train_model.py
- Loads audio files from Dataset/
- Extracts MFCC, Chroma, and Mel features
- Trains a deep neural network
- Saves model to models/ folder
- Takes 30-60 minutes to complete

### app.py
- Flask web server
- Handles file uploads
- Processes audio for prediction
- Returns emotion with confidence scores
- Serves the web interface

### templates/index.html
- Beautiful, responsive web UI
- Drag-and-drop file upload
- Real-time emotion analysis
- Animated results display
- Shows probability for all emotions

### setup_check.py
- Verifies Python version
- Checks installed packages
- Confirms dataset exists
- Validates trained models

## 🎓 Supported Emotions

| Emotion | Emoji | Code |
|---------|-------|------|
| Neutral | 😐 | 01 |
| Calm | 😌 | 02 |
| Happy | 😊 | 03 |
| Sad | 😢 | 04 |
| Angry | 😠 | 05 |
| Fearful | 😨 | 06 |
| Disgust | 🤢 | 07 |
| Surprised | 😲 | 08 |

## 🧠 Model Details

### Architecture
- Input: 168 features (40 MFCC + 12 Chroma + 128 Mel)
- 4 Dense layers with batch normalization
- Dropout for regularization (30%)
- Softmax output (8 classes)

### Training Configuration
- Optimizer: Adam
- Loss: Categorical Cross-entropy
- Batch Size: 32
- Epochs: Up to 100 (with early stopping)
- Validation Split: 20%

### Expected Performance
- Training Accuracy: 85-95%
- Validation Accuracy: 70-80%
- Test Accuracy: 65-75%

## 📊 Dataset Information

### RAVDESS Dataset
- **Name**: Ryerson Audio-Visual Database of Emotional Speech and Song
- **Actors**: 24 professional actors (12 male, 12 female)
- **Emotions**: 8 emotions
- **Total Files**: ~1,440 audio files
- **Format**: WAV (16-bit, 48kHz)

### Filename Convention
Format: `MM-VV-EE-SS-RR-AA-GG.wav`
- MM: Modality (03 = audio-video)
- VV: Vocal channel (01 = speech)
- EE: Emotion (01-08)
- SS: Statement (01-02)
- RR: Repetition (01-02)
- AA: Actor (01-24)
- GG: Gender (01 = male, 02 = female)

## 🌐 Web Interface Features

### Upload Options
- Drag and drop
- Click to browse
- Supported formats: WAV, MP3, OGG, FLAC
- Max file size: 16MB

### Results Display
- Primary emotion with emoji
- Confidence percentage
- Probability bars for all emotions
- Color-coded visualization

### User Experience
- Responsive design
- Loading animations
- Error handling
- Beautiful gradient theme

## 🔧 Technical Stack

### Backend
- **Framework**: Flask 3.0
- **ML Framework**: TensorFlow 2.15
- **Audio Processing**: Librosa 0.10
- **Data Processing**: NumPy, Pandas, scikit-learn

### Frontend
- **HTML5** with semantic markup
- **CSS3** with animations
- **Vanilla JavaScript** (no frameworks)
- **Responsive design**

## 📈 Training Process Explained

1. **Data Loading** (2-5 minutes)
   - Scans all Actor_XX folders
   - Reads WAV files
   - Extracts emotion labels from filenames

2. **Feature Extraction** (10-15 minutes)
   - Loads each audio file
   - Computes MFCC (40 features)
   - Computes Chroma (12 features)
   - Computes Mel spectrogram (128 features)
   - Total: 180 features per file

3. **Model Training** (15-40 minutes)
   - Splits data 80/20
   - Trains neural network
   - Validates on test set
   - Saves best model

4. **Model Saving** (1 minute)
   - Saves final model
   - Saves best checkpoint
   - Saves label encoder
   - Saves training history

## 🚨 Common Issues & Solutions

### Issue: Dependencies installation fails
**Solution**: 
```bash
python -m pip install --upgrade pip
pip install --upgrade setuptools wheel
pip install -r requirements.txt
```

### Issue: TensorFlow installation error on Windows
**Solution**: 
- Install Visual C++ Redistributable
- Use: `pip install tensorflow-cpu` instead

### Issue: librosa fails to load audio
**Solution**:
- Install ffmpeg
- Or convert audio to WAV format

### Issue: Model training is very slow
**Solution**:
- Training on CPU is normal (30-60 min)
- For GPU: Install tensorflow-gpu
- Reduce dataset size for testing

### Issue: Web app shows "Models not found"
**Solution**:
- Run `python train_model.py` first
- Wait for training to complete
- Check that `models/best_model.h5` exists

## 💡 Tips & Best Practices

1. **First Time Setup**
   - Run `check.bat` to verify everything
   - Install dependencies in a virtual environment
   - Ensure you have 2GB free disk space

2. **Training**
   - Don't close the terminal during training
   - Monitor the accuracy - should improve over epochs
   - The best model is saved automatically

3. **Using the Web App**
   - Use clear speech audio for best results
   - Audio should be 1-5 seconds long
   - WAV format works best

4. **Improving Accuracy**
   - Add more training data
   - Increase training epochs
   - Fine-tune model architecture

## 🎯 Next Steps After Setup

1. ✅ Verify installation with `check.bat`
2. ✅ Train model with `train.bat` (wait 30-60 min)
3. ✅ Test with your own audio files
4. ✅ Share with friends!

## 🔮 Future Enhancements Ideas

- [ ] Real-time microphone input
- [ ] Batch processing multiple files
- [ ] Export results to CSV
- [ ] Emotion trend charts
- [ ] REST API for integration
- [ ] Docker containerization
- [ ] Mobile app version

## 📚 Learning Resources

### Understanding the Model
- [MFCC Tutorial](https://librosa.org/doc/main/feature.html)
- [Deep Learning for Audio](https://towardsdatascience.com/audio-deep-learning-made-simple-part-1-state-of-the-art-techniques-da1d3dff2504)

### Dataset
- [RAVDESS Dataset](https://zenodo.org/record/1188976)

### Technologies Used
- [TensorFlow Docs](https://www.tensorflow.org/)
- [Librosa Docs](https://librosa.org/)
- [Flask Docs](https://flask.palletsprojects.com/)

## 📞 Support

If you encounter issues:
1. Check this file first
2. Review README.md
3. Run `python setup_check.py`
4. Check error messages carefully

## 🎉 Success Indicators

You'll know everything works when:
- ✅ `setup_check.py` shows all green checkmarks
- ✅ Training completes without errors
- ✅ Web app starts successfully
- ✅ You can upload and analyze audio files
- ✅ Results show emotion with confidence scores

## 📄 File Descriptions

| File | Purpose | When to Use |
|------|---------|-------------|
| install.bat | Install dependencies | First time setup |
| check.bat | Verify setup | Before training |
| train.bat | Train model | After installing deps |
| run.bat | Start web app | After training |
| train_model.py | Train via Python | Advanced users |
| app.py | Web server | Auto-run via run.bat |
| setup_check.py | Check status | Troubleshooting |

---

**🎭 You now have a complete, production-ready emotion detection system!**

**Ready to start? Run:** `install.bat` → `train.bat` → `run.bat`

---

*Last updated: January 2026*

"""
Visual Demo Script
Run this to see a visual representation of the project structure and status
"""

import os
import sys

def print_header():
    """Print a nice header"""
    print("\n" + "="*70)
    print("  🎭 SPEECH EMOTION RECOGNITION PROJECT 🎭")
    print("="*70 + "\n")

def print_tree():
    """Print project structure as a tree"""
    print("📁 Project Structure:\n")
    
    structure = """
Voice detector/
│
├── 🎯 Quick Start Files (Double-click these!)
│   ├── install.bat          ← Step 1: Install dependencies
│   ├── train.bat            ← Step 2: Train the model  
│   ├── run.bat              ← Step 3: Start web app
│   └── check.bat            ← Check your setup anytime
│
├── 🐍 Python Scripts
│   ├── train_model.py       ← AI model training script
│   ├── app.py               ← Flask web server
│   └── setup_check.py       ← Setup verification
│
├── 🌐 Web Interface
│   └── templates/
│       └── index.html       ← Beautiful web UI
│
├── 📊 Your Dataset
│   └── Dataset/
│       ├── Actor_01/        ← 24 actor folders
│       ├── Actor_02/        ← with WAV files
│       └── ...
│
├── 📚 Documentation
│   ├── README.md            ← Complete documentation
│   ├── QUICKSTART.md        ← Quick start guide
│   └── PROJECT_OVERVIEW.md  ← Detailed overview
│
└── ⚙️ Configuration
    └── requirements.txt     ← Python dependencies

After training, you'll have:
└── 💾 models/
    ├── emotion_model.h5     ← Trained AI model
    ├── best_model.h5        ← Best checkpoint
    └── label_encoder.pkl    ← Emotion labels
    """
    
    print(structure)

def print_workflow():
    """Print the workflow"""
    print("\n" + "="*70)
    print("🚀 SIMPLE 3-STEP WORKFLOW")
    print("="*70 + "\n")
    
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│  STEP 1: Install Dependencies                               │")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│  Double-click: install.bat                                  │")
    print("│  OR run: pip install -r requirements.txt                    │")
    print("│  Time: 5-10 minutes                                         │")
    print("└─────────────────────────────────────────────────────────────┘")
    print("")
    print("                          ⬇️")
    print("")
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│  STEP 2: Train the AI Model                                │")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│  Double-click: train.bat                                    │")
    print("│  OR run: python train_model.py                              │")
    print("│  Time: 30-60 minutes                                        │")
    print("│  What it does:                                              │")
    print("│   • Loads 1,440 audio files                                 │")
    print("│   • Extracts audio features                                 │")
    print("│   • Trains neural network                                   │")
    print("│   • Saves model to models/ folder                           │")
    print("└─────────────────────────────────────────────────────────────┘")
    print("")
    print("                          ⬇️")
    print("")
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│  STEP 3: Run the Web Application                           │")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│  Double-click: run.bat                                      │")
    print("│  OR run: python app.py                                      │")
    print("│  Then open: http://localhost:5000                           │")
    print("│                                                             │")
    print("│  What you can do:                                           │")
    print("│   • Upload audio files                                      │")
    print("│   • Get instant emotion predictions                         │")
    print("│   • See confidence scores                                   │")
    print("│   • View probability distribution                           │")
    print("└─────────────────────────────────────────────────────────────┘")
    print("")

def print_emotions():
    """Print supported emotions"""
    print("="*70)
    print("🎭 SUPPORTED EMOTIONS")
    print("="*70 + "\n")
    
    emotions = [
        ("😐 Neutral", "Calm, emotionless speech"),
        ("😌 Calm", "Peaceful, relaxed tone"),
        ("😊 Happy", "Joyful, excited expression"),
        ("😢 Sad", "Sorrowful, melancholic tone"),
        ("😠 Angry", "Aggressive, frustrated speech"),
        ("😨 Fearful", "Scared, anxious expression"),
        ("🤢 Disgust", "Repulsed, disgusted tone"),
        ("😲 Surprised", "Shocked, amazed expression")
    ]
    
    for emotion, description in emotions:
        print(f"  {emotion:20} → {description}")
    
    print("")

def print_features():
    """Print key features"""
    print("="*70)
    print("✨ KEY FEATURES")
    print("="*70 + "\n")
    
    features = [
        "🤖 Deep Learning AI Model",
        "🎵 Advanced Audio Processing (MFCC, Chroma, Mel)",
        "🌐 Beautiful Web Interface",
        "📊 Real-time Predictions",
        "📈 Confidence Scores & Probability Distribution",
        "🎯 8 Emotion Categories",
        "📁 Support for WAV, MP3, OGG, FLAC",
        "⚡ Fast Processing (<2 seconds)",
        "📱 Responsive Design",
        "🔒 Secure Local Processing"
    ]
    
    for feature in features:
        print(f"  ✓ {feature}")
    
    print("")

def check_files():
    """Check which files exist"""
    print("="*70)
    print("📋 FILE STATUS CHECK")
    print("="*70 + "\n")
    
    files_to_check = {
        'Core Scripts': ['train_model.py', 'app.py', 'setup_check.py'],
        'Batch Files': ['install.bat', 'train.bat', 'run.bat', 'check.bat'],
        'Documentation': ['README.md', 'QUICKSTART.md', 'PROJECT_OVERVIEW.md'],
        'Configuration': ['requirements.txt'],
        'Web Interface': ['templates/index.html'],
        'Dataset': ['Dataset/'],
        'Models': ['models/best_model.h5', 'models/label_encoder.pkl']
    }
    
    for category, files in files_to_check.items():
        print(f"\n{category}:")
        for file in files:
            exists = os.path.exists(file)
            status = "✅" if exists else "❌"
            note = ""
            
            if not exists and category == "Models":
                note = " (will be created after training)"
            
            print(f"  {status} {file}{note}")
    
    print("")

def print_next_steps():
    """Print next steps"""
    print("="*70)
    print("🎯 WHAT TO DO NEXT?")
    print("="*70 + "\n")
    
    if not os.path.exists('models'):
        print("You haven't trained the model yet! Here's what to do:\n")
        print("  1️⃣  Double-click 'install.bat' to install dependencies")
        print("  2️⃣  Double-click 'check.bat' to verify setup")
        print("  3️⃣  Double-click 'train.bat' to train the model (wait 30-60 min)")
        print("  4️⃣  Double-click 'run.bat' to start the web app")
        print("  5️⃣  Open http://localhost:5000 in your browser")
        print("  6️⃣  Upload audio and see emotions detected! 🎉")
    elif not os.path.exists('models/best_model.h5'):
        print("Models folder exists but training incomplete!\n")
        print("  → Run 'train.bat' to complete training")
    else:
        print("Model is trained! You're ready to go! 🎉\n")
        print("  → Run 'run.bat' to start the web application")
        print("  → Open http://localhost:5000 in your browser")
        print("  → Upload audio files and detect emotions!")
    
    print("\n" + "="*70)
    print("💡 TIP: Run 'check.bat' anytime to verify your setup!")
    print("="*70 + "\n")

def main():
    """Main function"""
    print_header()
    print_tree()
    print_workflow()
    print_emotions()
    print_features()
    check_files()
    print_next_steps()
    
    print("="*70)
    print("  📚 For detailed help, see README.md or QUICKSTART.md")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
    input("\nPress Enter to exit...")

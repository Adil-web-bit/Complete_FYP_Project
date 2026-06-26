"""
Quick Start Script for Speech Emotion Recognition Project
Run this script to check your setup and get started
"""

import sys
import os

def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    print("\nChecking dependencies...")
    required_packages = [
        'tensorflow',
        'librosa',
        'flask',
        'numpy',
        'pandas',
        'sklearn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT INSTALLED")
            missing_packages.append(package)
    
    if missing_packages:
        print("\n⚠️ Missing packages detected!")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def check_dataset():
    """Check if dataset exists"""
    print("\nChecking dataset...")
    if not os.path.exists('Dataset'):
        print("❌ Dataset folder not found!")
        return False
    
    actor_dirs = [d for d in os.listdir('Dataset') if d.startswith('Actor_')]
    if len(actor_dirs) == 0:
        print("❌ No actor directories found in Dataset folder!")
        return False
    
    print(f"✅ Found {len(actor_dirs)} actor directories")
    return True

def check_models():
    """Check if models are trained"""
    print("\nChecking trained models...")
    if not os.path.exists('models'):
        print("⚠️ Models folder not found - you need to train the model first")
        return False
    
    required_files = ['best_model.h5', 'label_encoder.pkl']
    missing_files = []
    
    for file in required_files:
        if os.path.exists(f'models/{file}'):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - NOT FOUND")
            missing_files.append(file)
    
    if missing_files:
        print("\n⚠️ Models not trained yet!")
        print("Run: python train_model.py")
        return False
    
    return True

def main():
    """Main setup check"""
    print("="*60)
    print("🎭 Speech Emotion Recognition - Setup Check")
    print("="*60)
    
    checks = {
        'Python Version': check_python_version(),
        'Dependencies': check_dependencies(),
        'Dataset': check_dataset(),
        'Trained Models': check_models()
    }
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_passed = all(checks.values())
    
    for check, status in checks.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {check}")
    
    print("\n" + "="*60)
    
    if all_passed:
        print("✅ ALL CHECKS PASSED!")
        print("\n🚀 You're ready to go!")
        print("\nTo start the web application, run:")
        print("   python app.py")
    else:
        print("⚠️ SOME CHECKS FAILED")
        print("\n📋 Next Steps:")
        
        if not checks['Dependencies']:
            print("1. Install dependencies: pip install -r requirements.txt")
        
        if not checks['Dataset']:
            print("2. Ensure the Dataset folder contains actor audio files")
        
        if not checks['Trained Models']:
            print("3. Train the model: python train_model.py")
        
        print("\nThen run this script again to verify.")
    
    print("="*60)

if __name__ == '__main__':
    main()

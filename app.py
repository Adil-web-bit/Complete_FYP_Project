"""
Human Behaviour Predictor - Home Page
AI-powered emotion recognition system
"""

import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Human Behaviour Predictor",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
    }
    
    .hero-section {
        text-align: center;
        padding: 4rem 2rem;
        color: white;
    }
    
    .main-title {
        font-size: 4rem;
        font-weight: 800;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        line-height: 1.2;
    }
    
    .subtitle {
        font-size: 1.5rem;
        margin-bottom: 1rem;
        opacity: 0.95;
        font-weight: 300;
    }
    
    .university {
        font-size: 1.1rem;
        opacity: 0.85;
        margin-top: 0.5rem;
    }
    
    .content-section {
        background: white;
        border-radius: 30px 30px 0 0;
        padding: 3rem 2rem;
        margin-top: 2rem;
    }
    
    .feature-card {
        background: white;
        border: 2px solid #e2e8f0;
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        transition: all 0.3s ease;
        height: 100%;
        cursor: pointer;
    }
    
    .feature-card:hover {
        border-color: #667eea;
        box-shadow: 0 15px 30px rgba(102, 126, 234, 0.2);
        transform: translateY(-8px);
    }
    
    .feature-icon {
        font-size: 5rem;
        margin-bottom: 1.5rem;
        display: block;
    }
    
    .feature-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2d3748;
        margin-bottom: 1rem;
    }
    
    .feature-desc {
        color: #718096;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    
    .stats-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
    }
    
    .stat-number {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    
    .tech-badge {
        display: inline-block;
        background: #edf2f7;
        color: #2d3748;
        padding: 0.6rem 1.3rem;
        border-radius: 25px;
        font-size: 0.95rem;
        font-weight: 600;
        margin: 0.4rem;
    }
    
    .cta-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 3rem;
        border-radius: 50px;
        font-size: 1.2rem;
        font-weight: 700;
        border: none;
        cursor: pointer;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .cta-button:hover {
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.4);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero-section">
    <h1 class="main-title">🎭 Human Behaviour Predictor</h1>
    <p class="subtitle">Advanced AI-Powered Emotion Recognition System</p>
    <p style="font-size: 1.1rem; opacity: 0.9;">Analyzing human emotions through facial expressions and voice patterns using deep learning</p>
    <p class="university">🎓 University of Engineering and Technology, Taxila | Final Year Project 2025-2026</p>
</div>
""", unsafe_allow_html=True)

# Content Section
st.markdown('<div class="content-section">', unsafe_allow_html=True)

# Project Overview
st.markdown("<h2 style='text-align: center; color: #2d3748; font-weight: 700; margin-bottom: 3rem; font-size: 2.5rem;'>🚀 What We Offer</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    <div class="feature-card">
        <span class="feature-icon">📸</span>
        <h3 class="feature-title">Facial Expression Analysis</h3>
        <p class="feature-desc">
            Advanced computer vision technology that detects and analyzes human emotions 
            from facial expressions using deep learning CNN models. Achieve high accuracy 
            with MTCNN face detection and real-time emotion classification.
        </p>
        <div style="margin-top: 2rem;">
            <div class="tech-badge">MTCNN</div>
            <div class="tech-badge">CNN</div>
            <div class="tech-badge">82% Accuracy</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <span class="feature-icon">🎤</span>
        <h3 class="feature-title">Voice Pattern Analysis</h3>
        <p class="feature-desc">
            Sophisticated audio processing that extracts emotional cues from voice recordings. 
            Using MFCC, chroma, and mel-spectrogram features, our neural network identifies 
            emotions with high precision from speech patterns.
        </p>
        <div style="margin-top: 2rem;">
            <div class="tech-badge">MFCC</div>
            <div class="tech-badge">Librosa</div>
            <div class="tech-badge">Neural Net</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Statistics
st.markdown("<br><h2 style='text-align: center; color: #2d3748; font-weight: 700; margin: 4rem 0 2rem 0; font-size: 2.5rem;'>📊 System Capabilities</h2>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="stats-card">
        <div class="stat-number">7</div>
        <div class="stat-label">Emotion Categories</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stats-card" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);">
        <div class="stat-number">2</div>
        <div class="stat-label">Detection Modes</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stats-card" style="background: linear-gradient(135deg, #fc5c7d 0%, #6a82fb 100%);">
        <div class="stat-number">82%</div>
        <div class="stat-label">Model Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="stats-card" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);">
        <div class="stat-number">&lt;2s</div>
        <div class="stat-label">Processing Time</div>
    </div>
    """, unsafe_allow_html=True)

# Technology Stack
st.markdown("<br><h2 style='text-align: center; color: #2d3748; font-weight: 700; margin: 4rem 0 2rem 0; font-size: 2.5rem;'>⚙️ Technology Stack</h2>", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; padding: 2rem; background: #f7fafc; border-radius: 20px;'>
    <span class='tech-badge' style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;'>Python</span>
    <span class='tech-badge' style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;'>TensorFlow</span>
    <span class='tech-badge' style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;'>Keras</span>
    <span class='tech-badge' style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;'>OpenCV</span>
    <span class='tech-badge' style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;'>Librosa</span>
    <span class='tech-badge' style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;'>Streamlit</span>
    <span class='tech-badge' style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;'>NumPy</span>
    <span class='tech-badge' style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;'>Plotly</span>
</div>
""", unsafe_allow_html=True)

# Call to Action
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; padding: 3rem 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
     border-radius: 20px; color: white;'>
    <h2 style='font-size: 2.5rem; font-weight: 800; margin-bottom: 1rem;'>Ready to Analyze Emotions?</h2>
    <p style='font-size: 1.2rem; margin-bottom: 2rem; opacity: 0.95;'>Choose a detection mode from the sidebar to get started!</p>
    <p style='font-size: 1rem; opacity: 0.85;'>👈 Select either Facial Expression Analysis or Voice Analysis from the navigation menu</p>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Custom CSS for professional styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(to bottom, #f8f9fa 0%, #ffffff 100%);
    }
    
    .main-header {
        font-size: 3rem;
        color: #1a1a2e;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #2d3748;
        margin-bottom: 1.5rem;
        font-weight: 600;
        border-left: 4px solid #4299e1;
        padding-left: 1rem;
    }
    
    .feature-card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        color: #2d3748;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        box-shadow: 0 10px 15px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
    }
    
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
    }
    
    .warning-card {
        background: linear-gradient(135deg, #fc5c7d 0%, #6a82fb 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 25px rgba(252, 92, 125, 0.3);
    }
    
    .upload-area {
        border: 2px dashed #cbd5e0;
        border-radius: 12px;
        padding: 2.5rem;
        text-align: center;
        background: white;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #4299e1;
        background: #f7fafc;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    .info-badge {
        display: inline-block;
        background: #edf2f7;
        color: #2d3748;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    .project-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        margin: 2rem 0 1rem 0;
        letter-spacing: -1px;
    }
    
    .subtitle {
        text-align: center;
        color: #718096;
        font-size: 1.25rem;
        margin-bottom: 3rem;
        font-weight: 400;
    }
    
    .mode-card {
        background: white;
        border: 2px solid #e2e8f0;
        border-radius: 16px;
        padding: 2.5rem;
        margin: 1rem;
        transition: all 0.3s ease;
        cursor: pointer;
        text-align: center;
    }
    
    .mode-card:hover {
        border-color: #667eea;
        box-shadow: 0 12px 24px rgba(102, 126, 234, 0.15);
        transform: translateY(-5px);
    }
    
    .mode-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)


# Load Lottie animation
def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None


# ==============================
# Model Loading Functions
# ==============================
# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_emotion_image_model():
    """Load the emotion detection model for images"""
    try:
        model_path = os.path.join(SCRIPT_DIR, 'Face gesture', 'emotion_detection_model.keras')
        return load_model(model_path)
    except:
        try:
            model_path = os.path.join(SCRIPT_DIR, 'Face gesture', 'best_emotion_model.h5')
            return load_model(model_path)
        except Exception as e:
            st.error(f"Could not load image emotion model: {e}")
            return None


@st.cache_resource
def load_face_detector():
    """Load MTCNN face detector"""
    return MTCNN()


@st.cache_resource
def load_voice_emotion_model():
    """Load the voice emotion detection model"""
    try:
        model_path = os.path.join(SCRIPT_DIR, 'Voice detector', 'models', 'best_model.h5')
        encoder_path = os.path.join(SCRIPT_DIR, 'Voice detector', 'models', 'label_encoder.pkl')
        
        # Custom Dense layer to ignore quantization_config
        from tensorflow.keras.layers import Dense as OriginalDense
        
        class CustomDense(OriginalDense):
            def __init__(self, *args, **kwargs):
                # Remove quantization_config if present
                kwargs.pop('quantization_config', None)
                super().__init__(*args, **kwargs)
        
        # Load model with custom objects
        custom_objects = {'Dense': CustomDense}
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        
        with open(encoder_path, 'rb') as f:
            encoder = pickle.load(f)
        return model, encoder
    except Exception as e:
        st.error(f"Could not load voice emotion model: {e}")
        return None, None


# ==============================
# Image Emotion Detection Functions
# ==============================
def detect_and_crop_face(image_array):
    """
    Detect face in image and return cropped face
    Returns: (face_crop, bounding_box, confidence) or (None, None, 0) if no face
    """
    face_detector = load_face_detector()
    
    # Convert PIL to OpenCV format
    if isinstance(image_array, np.ndarray):
        rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    else:
        rgb_image = cv2.cvtColor(np.array(image_array), cv2.COLOR_BGR2RGB)

    # Detect faces
    faces = face_detector.detect_faces(rgb_image)

    if len(faces) == 0:
        return None, None, 0

    # Get the face with highest confidence
    faces.sort(key=lambda x: x['confidence'], reverse=True)
    best_face = faces[0]
    confidence = best_face['confidence']

    # Extract bounding box
    x, y, w, h = best_face['box']

    # Add padding (20% around face)
    padding_w = int(w * 0.2)
    padding_h = int(h * 0.2)

    # Ensure coordinates are within image bounds
    x1 = max(0, x - padding_w)
    y1 = max(0, y - padding_h)
    x2 = min(rgb_image.shape[1], x + w + padding_w)
    y2 = min(rgb_image.shape[0], y + h + padding_h)

    # Crop face
    face_crop = rgb_image[y1:y2, x1:x2]

    # Convert back to PIL
    face_pil = Image.fromarray(face_crop)

    # Draw bounding box on original image (for visualization)
    bbox_image = rgb_image.copy()
    cv2.rectangle(bbox_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    bbox_pil = Image.fromarray(bbox_image)

    return face_pil, bbox_pil, confidence


def preprocess_face_for_emotion(face_image):
    """Preprocess face image for emotion model"""
    # Convert to grayscale
    face_gray = face_image.convert('L')

    # Resize to 48x48 (as per model's expected input)
    face_resized = face_gray.resize((48, 48))

    # Convert to numpy array and normalize
    face_array = np.array(face_resized) / 255.0

    # Add batch and channel dimensions
    face_array = np.expand_dims(face_array, axis=0)
    face_array = np.expand_dims(face_array, axis=-1)

    return face_array


# ==============================
# Voice Emotion Detection Functions
# ==============================
def extract_audio_features(file_path):
    """Extract audio features from a file"""
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
        st.error(f"Error extracting features: {e}")
        return None


def predict_voice_emotion(file_path, model, label_encoder):
    """Predict emotion from audio file"""
    if model is None or label_encoder is None:
        return None
    
    # Extract features
    features = extract_audio_features(file_path)
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
        best_emotion = max(all_emotions.items(), key=lambda x: x[1])
        emotion = best_emotion[0]
        confidence = best_emotion[1]
    
    return {
        'emotion': emotion,
        'confidence': float(confidence),
        'all_probabilities': all_emotions
    }


# ==============================
# Initialize Session State
# ==============================
if 'detection_mode' not in st.session_state:
    st.session_state.detection_mode = None


# ==============================
# Main Application
# ==============================
st.markdown("""
<div style='text-align: center; padding: 2rem 0;'>
    <h1 class='project-title'>🎭 Human Behaviour Predictor</h1>
    <p class='subtitle'>
        Advanced AI-powered emotion recognition system analyzing facial expressions and voice patterns<br>
        <span style='color: #a0aec0; font-size: 0.95rem;'>Final Year Project | Deep Learning & Computer Vision</span>
    </p>
</div>
""", unsafe_allow_html=True)


# ==============================
# Mode Selection
# ==============================
if st.session_state.detection_mode is None:
    st.markdown('<h2 style="text-align: center; color: #2d3748; font-weight: 600; margin: 3rem 0 2rem 0;">Select Analysis Mode</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        if st.button("📸 Facial Expression Analysis", key="image_mode", use_container_width=True):
            st.session_state.detection_mode = "image"
            st.rerun()
        
        st.markdown("""
        <div class='feature-card'>
            <div style='text-align: center; margin-bottom: 1.5rem;'>
                <span style='font-size: 3.5rem;'>📷</span>
            </div>
            <h3 style='color: #2d3748; text-align: center; margin-bottom: 1rem; font-weight: 600;'>Image-based Detection</h3>
            <div style='color: #4a5568; line-height: 1.8;'>
                <p style='margin: 0.5rem 0;'>✓ Intelligent face detection using MTCNN</p>
                <p style='margin: 0.5rem 0;'>✓ 7 distinct emotion categories</p>
                <p style='margin: 0.5rem 0;'>✓ Deep learning CNN architecture</p>
                <p style='margin: 0.5rem 0;'>✓ Instant visual analysis</p>
                <p style='margin: 0.5rem 0;'>✓ High accuracy predictions</p>
            </div>
            <div style='margin-top: 1.5rem; text-align: center;'>
                <span class='info-badge'>MTCNN</span>
                <span class='info-badge'>TensorFlow</span>
                <span class='info-badge'>CNN</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("🎤 Voice Pattern Analysis", key="voice_mode", use_container_width=True):
            st.session_state.detection_mode = "voice"
            st.rerun()
        
        st.markdown("""
        <div class='feature-card'>
            <div style='text-align: center; margin-bottom: 1.5rem;'>
                <span style='font-size: 3.5rem;'>🎵</span>
            </div>
            <h3 style='color: #2d3748; text-align: center; margin-bottom: 1rem; font-weight: 600;'>Audio-based Detection</h3>
            <div style='color: #4a5568; line-height: 1.8;'>
                <p style='margin: 0.5rem 0;'>✓ Advanced audio feature extraction</p>
                <p style='margin: 0.5rem 0;'>✓ MFCC & spectral analysis</p>
                <p style='margin: 0.5rem 0;'>✓ Multiple format support</p>
                <p style='margin: 0.5rem 0;'>✓ RAVDESS dataset trained</p>
                <p style='margin: 0.5rem 0;'>✓ Real-time processing</p>
            </div>
            <div style='margin-top: 1.5rem; text-align: center;'>
                <span class='info-badge'>Librosa</span>
                <span class='info-badge'>Neural Net</span>
                <span class='info-badge'>MFCC</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Features section
    st.markdown("<hr style='margin: 3rem 0; border: none; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: #2d3748; font-weight: 600; margin-bottom: 2rem;">System Features</h2>', unsafe_allow_html=True)
    
    col3, col4, col5 = st.columns(3, gap="large")
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div style='text-align: center; font-size: 2.5rem; margin-bottom: 1rem;'>🎯</div>
            <h3 style='color: #2d3748; text-align: center; font-weight: 600; margin-bottom: 0.75rem;'>Dual Analysis</h3>
            <p style='color: #718096; text-align: center; line-height: 1.6;'>Comprehensive emotion detection through both facial expressions and voice patterns</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="feature-card">
            <div style='text-align: center; font-size: 2.5rem; margin-bottom: 1rem;'>⚡</div>
            <h3 style='color: #2d3748; text-align: center; font-weight: 600; margin-bottom: 0.75rem;'>Real-time Processing</h3>
            <p style='color: #718096; text-align: center; line-height: 1.6;'>Instant results powered by optimized deep learning models</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown("""
        <div class="feature-card">
            <div style='text-align: center; font-size: 2.5rem; margin-bottom: 1rem;'>🔒</div>
            <h3 style='color: #2d3748; text-align: center; font-weight: 600; margin-bottom: 0.75rem;'>Privacy First</h3>
            <p style='color: #718096; text-align: center; line-height: 1.6;'>Secure processing with no data storage or external transmission</p>
        </div>
        """, unsafe_allow_html=True)

else:
    # Back button
    if st.button("← Back to Mode Selection"):
        st.session_state.detection_mode = None
        st.rerun()
    
    # ==============================
    # IMAGE EMOTION DETECTION MODE
    # ==============================
    if st.session_state.detection_mode == "image":
        st.markdown('<div class="sub-header" style="text-align: center;">📸 Image Emotion Detection</div>', unsafe_allow_html=True)
        
        # Load models
        image_model = load_emotion_image_model()
        
        if image_model is None:
            st.error("Could not load the image emotion detection model. Please check the model files.")
        else:
            # Emotion labels and colors
            emotion_labels = ['Angry 😠', 'Disgust 🤢', 'Fear 😨', 'Happy 😊', 'Neutral 😐', 'Sad 😢', 'Surprise 😲']
            emotion_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFE66D', '#95A5A6', '#5C6BC0', '#FFA726']
            emotion_emojis = ['😠', '🤢', '😨', '😊', '😐', '😢', '😲']
            
            # Sidebar
            with st.sidebar:
                st.markdown("## 🎯 About This AI")
                st.markdown("""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 15px; color: white;'>
                <h4 style='color: white; margin-bottom: 1rem;'>🤖 Model Specifications</h4>
                <p><strong>Face Detection:</strong> MTCNN</p>
                <p><strong>Emotion Model:</strong> CNN</p>
                <p><strong>Classes:</strong> 7 Emotions</p>
                <p><strong>Accuracy:</strong> 82%</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown("### ⚙️ Detection Settings")
                confidence_threshold = st.slider(
                    "Face Detection Confidence",
                    min_value=0.5,
                    max_value=0.99,
                    value=0.85,
                    step=0.01,
                    help="Higher values = more strict face detection"
                )
            
            # Main content
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown('<div class="sub-header">📤 Upload Your Image</div>', unsafe_allow_html=True)
                
                st.markdown("""
                <div class="upload-area">
                    <h3 style='color: #2E86AB;'>Drag & Drop or Click to Upload</h3>
                    <p style='color: #666;'>Upload any image - AI will detect faces automatically</p>
                </div>
                """, unsafe_allow_html=True)
                
                uploaded_file = st.file_uploader(
                    " ",
                    type=['jpg', 'jpeg', 'png', 'bmp'],
                    help="Upload any image with or without faces",
                    label_visibility="collapsed"
                )
                
                if uploaded_file is not None:
                    try:
                        # Load image
                        image = Image.open(uploaded_file)
                        
                        # Convert to RGB if needed
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        
                        # Display original image
                        st.markdown("### 📷 Original Image")
                        
                        # Convert to base64 for HTML display
                        buffer = BytesIO()
                        image.save(buffer, format="PNG")
                        img_str = base64.b64encode(buffer.getvalue()).decode()
                        
                        st.markdown(f"""
                        <div style='text-align: center;'>
                            <img src='data:image/png;base64,{img_str}'
                                 style='max-width:400px; max-height:400px;
                                        width:auto; height:auto;
                                        border-radius:10px; object-fit:contain; 
                                        box-shadow:0 4px 10px rgba(0,0,0,0.1);'>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Convert PIL to numpy for face detection
                        image_np = np.array(image)
                    
                    except Exception as e:
                        st.error(f"Error loading image: {str(e)}")
                        image_np = None
            
            with col2:
                st.markdown('<div class="sub-header">🔍 Analysis Results</div>', unsafe_allow_html=True)
                
                if uploaded_file is not None and image_np is not None:
                    with st.spinner('🔍 Detecting faces in the image...'):
                        progress_bar = st.progress(0)
                        
                        # Detect face
                        face_crop, bbox_image, face_confidence = detect_and_crop_face(image_np)
                        progress_bar.progress(100)
                        
                        if face_crop is None:
                            # No face detected
                            st.markdown(f"""
                            <div class="warning-card">
                                <h2 style='margin: 0; font-size: 3rem;'>🚫</h2>
                                <h3 style='margin: 0.5rem 0;'>No Face Detected</h3>
                                <h1 style='margin: 0; font-size: 2rem;'>NO HUMAN FACE FOUND</h1>
                                <div style='font-size: 1.2rem; margin-top: 1rem;'>
                                    Please upload an image with a clear human face
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.info("""
                            **Tips for better face detection:**
                            - Use clear, well-lit images
                            - Face should be visible and not too small
                            - Avoid extreme angles or covered faces
                            """)
                        
                        else:
                            # Face detected
                            st.markdown("### ✅ Face Detected!")
                            
                            # Display image with bounding box
                            bbox_buffer = BytesIO()
                            bbox_image.save(bbox_buffer, format="PNG")
                            bbox_str = base64.b64encode(bbox_buffer.getvalue()).decode()
                            
                            st.markdown(f"""
                            <div style='text-align: center;'>
                                <h4 style='color: #2E86AB;'>Face Detection Confidence: {face_confidence:.1%}</h4>
                                <img src='data:image/png;base64,{bbox_str}'
                                     style='max-width:300px; max-height:300px;
                                            width:auto; height:auto;
                                            border-radius:10px; object-fit:contain; 
                                            border: 3px solid #2E86AB;'>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Analyze emotion
                            with st.spinner('🤖 Analyzing emotion...'):
                                processed_face = preprocess_face_for_emotion(face_crop)
                                predictions = image_model.predict(processed_face, verbose=0)
                                
                                emotion_index = np.argmax(predictions[0])
                                confidence = np.max(predictions[0])
                                all_confidences = predictions[0]
                                
                                # Result Card
                                st.markdown(f"""
                                <div class="result-card">
                                    <h2 style='margin: 0; font-size: 2.5rem;'>{emotion_emojis[emotion_index]}</h2>
                                    <h3 style='margin: 0.5rem 0;'>Detected Emotion</h3>
                                    <h1 style='margin: 0; font-size: 2rem;'>{emotion_labels[emotion_index]}</h1>
                                    <div style='font-size: 1.5rem; margin-top: 1rem;'>
                                        Confidence: <strong>{confidence:.2%}</strong>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Confidence Chart
                                st.markdown("### 📊 Confidence Distribution")
                                fig = go.Figure(data=[go.Pie(
                                    labels=emotion_labels,
                                    values=all_confidences,
                                    hole=.4,
                                    marker_colors=emotion_colors,
                                    textinfo='label+percent',
                                    insidetextorientation='radial'
                                )])
                                fig.update_layout(showlegend=False, height=400, margin=dict(t=0, b=0, l=0, r=0))
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Detailed scores
                                st.markdown("### 🎯 Detailed Scores")
                                for label, conf, color in zip(emotion_labels, all_confidences, emotion_colors):
                                    col_a, col_b, col_c = st.columns([2, 5, 1])
                                    with col_a:
                                        st.markdown(f"**{label}**")
                                    with col_b:
                                        st.progress(float(conf))
                                    with col_c:
                                        st.markdown(f"**{conf:.1%}**")
    
    # ==============================
    # VOICE EMOTION DETECTION MODE
    # ==============================
    elif st.session_state.detection_mode == "voice":
        st.markdown('<div class="sub-header" style="text-align: center;">🎤 Voice Emotion Detection</div>', unsafe_allow_html=True)
        
        # Load models
        voice_model, label_encoder = load_voice_emotion_model()
        
        if voice_model is None or label_encoder is None:
            st.error("Could not load the voice emotion detection model. Please check the model files.")
        else:
            # Emotion mapping for voice
            emotion_emoji_map = {
                'angry': '😠',
                'disgust': '🤢',
                'fearful': '😨',
                'happy': '😊',
                'sad': '😢',
                'neutral': '😐',
                'surprised': '😲'
            }
            
            # Sidebar
            with st.sidebar:
                st.markdown("## 🎯 About This AI")
                st.markdown("""
                <div style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); padding: 1.5rem; border-radius: 15px; color: white;'>
                <h4 style='color: white; margin-bottom: 1rem;'>🤖 Model Specifications</h4>
                <p><strong>Model Type:</strong> Neural Network</p>
                <p><strong>Features:</strong> MFCC, Chroma, Mel</p>
                <p><strong>Dataset:</strong> RAVDESS</p>
                <p><strong>Supported:</strong> WAV, MP3, OGG</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown("### 💡 How It Works")
                steps = [
                    "1️⃣ Upload audio file",
                    "2️⃣ Extract audio features",
                    "3️⃣ Analyze emotion patterns",
                    "4️⃣ Get confidence scores"
                ]
                for step in steps:
                    st.markdown(f"{step}")
            
            # Main content
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown('<div class="sub-header">📤 Upload Audio File</div>', unsafe_allow_html=True)
                
                st.markdown("""
                <div class="upload-area">
                    <h3 style='color: #2E86AB;'>🎵 Drag & Drop or Click to Upload</h3>
                    <p style='color: #666;'>Upload WAV, MP3, OGG, or FLAC files</p>
                </div>
                """, unsafe_allow_html=True)
                
                uploaded_audio = st.file_uploader(
                    " ",
                    type=['wav', 'mp3', 'ogg', 'flac'],
                    help="Upload audio file for emotion detection",
                    label_visibility="collapsed"
                )
                
                if uploaded_audio is not None:
                    st.markdown("### 🎵 Uploaded Audio")
                    st.audio(uploaded_audio, format=f'audio/{uploaded_audio.name.split(".")[-1]}')
                    
                    # File info
                    file_size = uploaded_audio.size / 1024
                    st.info(f"📁 **File:** {uploaded_audio.name} ({file_size:.2f} KB)")
            
            with col2:
                st.markdown('<div class="sub-header">🔍 Analysis Results</div>', unsafe_allow_html=True)
                
                if uploaded_audio is not None:
                    with st.spinner('🎵 Analyzing audio...'):
                        # Save temporary file with absolute path
                        temp_path = os.path.join(SCRIPT_DIR, f"temp_audio.{uploaded_audio.name.split('.')[-1]}")
                        with open(temp_path, 'wb') as f:
                            f.write(uploaded_audio.read())
                        
                        # Predict emotion
                        result = predict_voice_emotion(temp_path, voice_model, label_encoder)
                        
                        # Clean up
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        
                        if result is None:
                            st.error("Error processing audio file. Please try another file.")
                        else:
                            emotion = result['emotion']
                            confidence = result['confidence']
                            all_probs = result['all_probabilities']
                            
                            # Result Card
                            emoji = emotion_emoji_map.get(emotion.lower(), '😐')
                            st.markdown(f"""
                            <div class="result-card">
                                <h2 style='margin: 0; font-size: 2.5rem;'>{emoji}</h2>
                                <h3 style='margin: 0.5rem 0;'>Detected Emotion</h3>
                                <h1 style='margin: 0; font-size: 2rem;'>{emotion.upper()}</h1>
                                <div style='font-size: 1.5rem; margin-top: 1rem;'>
                                    Confidence: <strong>{confidence:.2f}%</strong>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Probability chart
                            st.markdown("### 📊 Emotion Probabilities")
                            
                            # Sort by probability
                            sorted_emotions = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
                            
                            emotion_names = [e[0].capitalize() for e in sorted_emotions]
                            probabilities = [e[1] for e in sorted_emotions]
                            
                            fig = go.Figure(data=[
                                go.Bar(
                                    x=probabilities,
                                    y=emotion_names,
                                    orientation='h',
                                    marker=dict(
                                        color=probabilities,
                                        colorscale='Viridis',
                                        showscale=False
                                    ),
                                    text=[f'{p:.1f}%' for p in probabilities],
                                    textposition='auto',
                                )
                            ])
                            
                            fig.update_layout(
                                xaxis_title="Probability (%)",
                                yaxis_title="Emotion",
                                height=400,
                                margin=dict(l=0, r=0, t=0, b=0)
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Detailed scores
                            st.markdown("### 🎯 Detailed Scores")
                            for emotion_name, prob in sorted_emotions:
                                col_a, col_b, col_c = st.columns([2, 5, 1])
                                with col_a:
                                    emoji = emotion_emoji_map.get(emotion_name.lower(), '😐')
                                    st.markdown(f"**{emoji} {emotion_name.capitalize()}**")
                                with col_b:
                                    st.progress(float(prob / 100))
                                with col_c:
                                    st.markdown(f"**{prob:.1f}%**")

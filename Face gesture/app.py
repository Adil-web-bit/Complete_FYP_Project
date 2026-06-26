import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from streamlit_lottie import st_lottie
import requests
import json
import base64
from io import BytesIO
from mtcnn import MTCNN
import warnings

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="AI Emotion Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1, #FFE66D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2E86AB;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 0.5rem 0;
    }
    .confidence-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2E86AB;
        margin: 0.5rem 0;
    }
    .result-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .warning-card {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .upload-area {
        border: 2px dashed #2E86AB;
        border-radius: 15px;
        padding: 3rem;
        text-align: center;
        background: #f8f9fa;
        margin: 1rem 0;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1, #FFE66D);
    }
    .face-box {
        border: 3px solid #2E86AB;
        border-radius: 10px;
        padding: 5px;
        display: inline-block;
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


lottie_ai = load_lottie_url("https://assets1.lottiefiles.com/packages/lf20_5tkzkblw.json")


# Load the trained model
@st.cache_resource
def load_emotion_model():
    return load_model('emotion_detection_model.keras')


@st.cache_resource
def load_face_detector():
    return MTCNN()


model = load_emotion_model()
face_detector = load_face_detector()

# Emotion labels and colors
emotion_labels = ['Angry 😠', 'Disgust 🤢', 'Fear 😨', 'Happy 😊', 'Neutral 😐', 'Sad 😢', 'Surprise 😲']
emotion_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFE66D', '#95A5A6', '#5C6BC0', '#FFA726']
emotion_emojis = ['😠', '🤢', '😨', '😊', '😐', '😢', '😲']


def detect_and_crop_face(image_array):
    """
    Detect face in image and return cropped face
    Returns: (face_crop, bounding_box, confidence) or (None, None, 0) if no face
    """
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

    # Resize to 48x48 (as per your model's expected input)
    face_resized = face_gray.resize((48, 48))

    # Convert to numpy array and normalize
    face_array = np.array(face_resized) / 255.0

    # Add batch and channel dimensions
    face_array = np.expand_dims(face_array, axis=0)  # Add batch dimension
    face_array = np.expand_dims(face_array, axis=-1)  # Add channel dimension

    return face_array


# ==============================
# App title and description
# ==============================
st.markdown("""
<div style='text-align: center; margin-top: 2rem; margin-bottom: 3rem;'>
    <h1 style='font-size: 3.5rem; 
               background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1, #FFE66D);
               -webkit-background-clip: text;
               -webkit-text-fill-color: transparent;
               font-weight: bold;'>
        🎭 AI Emotion Detector
    </h1>
    <p style='font-size: 1.3rem; color: #555; margin-top: 1rem;'>
        Upload a facial image and let the AI detect the emotion! <br>
        <strong>Now with face detection - only detects emotions when faces are found!</strong>
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar for additional info
with st.sidebar:
    st.markdown("## 🎯 About This AI")

    if lottie_ai:
        st_lottie(lottie_ai, height=150, key="ai_animation")

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

    st.markdown("### 💡 How It Works")
    steps = [
        "1️⃣ Upload any image",
        "2️⃣ AI detects if a face is present",
        "3️⃣ If face found, it's automatically cropped",
        "4️⃣ Emotion is analyzed from the cropped face",
        "5️⃣ Get detailed results with confidence scores"
    ]

    for step in steps:
        st.markdown(f"{step}")

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

# ==============================
# Main content area
# ==============================
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

            # Step 1: Detect face
            face_crop, bbox_image, face_confidence = detect_and_crop_face(image_np)

            # Simulate progress
            for i in range(100):
                pass
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
                - Try a different image with better visibility
                """)

            else:
                # Face detected - show bounding box
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
                                border: 3px solid #2E86AB;
                                box-shadow:0 4px 10px rgba(0,0,0,0.1);'>
                    <p style='color:#555;'>Green box shows detected face</p>
                </div>
                """, unsafe_allow_html=True)

                # Display cropped face
                st.markdown("### 👤 Cropped Face for Analysis")
                face_buffer = BytesIO()
                face_crop.save(face_buffer, format="PNG")
                face_str = base64.b64encode(face_buffer.getvalue()).decode()

                st.markdown(f"""
                <div style='text-align: center;'>
                    <img src='data:image/png;base64,{face_str}'
                         style='max-width:200px; max-height:200px;
                                width:auto; height:auto;
                                border-radius:50%; object-fit:cover; 
                                box-shadow:0 4px 10px rgba(0,0,0,0.1);'>
                    <p style='color:#555;'>This cropped face will be analyzed</p>
                </div>
                """, unsafe_allow_html=True)

                # Check face detection confidence
                if face_confidence < confidence_threshold:
                    st.warning(
                        f"⚠️ Face detection confidence ({face_confidence:.1%}) is below threshold ({confidence_threshold:.0%}). Results may be less accurate.")

                # Analyze emotion
                with st.spinner('🤖 Analyzing emotion from face...'):
                    progress_bar2 = st.progress(0)

                    # Preprocess face for emotion model
                    processed_face = preprocess_face_for_emotion(face_crop)

                    # Predict emotion
                    predictions = model.predict(processed_face, verbose=0)

                    # Simulate progress
                    for i in range(100):
                        pass
                    progress_bar2.progress(100)

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

                    # Confidence Scores
                    st.markdown("### 🎯 Detailed Confidence Scores")
                    for i, (label, conf, color) in enumerate(zip(emotion_labels, all_confidences, emotion_colors)):
                        col_a, col_b, col_c = st.columns([2, 5, 1])
                        with col_a:
                            st.markdown(f"**{label}**")
                        with col_b:
                            st.progress(float(conf))
                        with col_c:
                            st.markdown(f"**{conf:.1%}**")

# ==============================
# Features section (unchanged)
# ==============================
st.markdown("---")
st.markdown('<div class="sub-header" style="text-align: center;">🚀 Why Choose Our AI?</div>', unsafe_allow_html=True)

col3, col4, col5 = st.columns(3)

with col3:
    st.markdown("""
    <div class="feature-card">
        <h3>🎯 Smart Face Detection</h3>
        <p>Automatically detects faces before emotion analysis</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-card" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);">
        <h3>⚡ Real-time Analysis</h3>
        <p>Get instant results with our optimized deep learning model</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="feature-card" style="background: linear-gradient(135deg, #fc466b 0%, #3f5efb 100%);">
        <h3>🔍 7 Unique Emotions</h3>
        <p>Comprehensive detection across the full spectrum of human emotions</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-card" style="background: linear-gradient(135deg, #fdbb2d 0%, #22c1c3 100%);">
        <h3>📈 Detailed Analytics</h3>
        <p>Visual confidence scores and detailed emotion breakdown</p>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div class="feature-card" style="background: linear-gradient(135deg, #833ab4 0%, #fd1d1d 50%, #fcb045 100%);">
        <h3>🎨 User-Friendly Model</h3>
        <p>Beautiful interface designed for seamless user experience</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-card" style="background: linear-gradient(135deg, #00b4db 0%, #0083b0 100%);">
        <h3>🔒 Privacy First</h3>
        <p>Your images are processed securely and never stored</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================
# Tech stack (updated)
# ==============================
st.markdown("---")
st.markdown('<div class="sub-header" style="text-align: center;">🛠️ Powered By</div>', unsafe_allow_html=True)

tech_cols = st.columns(5)
tech_icons = ["🤖", "🔍", "📚", "🎯", "⚡"]
tech_names = ["TensorFlow", "MTCNN", "Keras", "CNN", "Streamlit"]

for i, (icon, name) in enumerate(zip(tech_icons, tech_names)):
    with tech_cols[i]:
        st.markdown(f"""
        <div style='text-align: center; padding: 1rem;'>
            <div style='font-size: 3rem;'>{icon}</div>
            <h4>{name}</h4>
        </div>
        """, unsafe_allow_html=True)

# ==============================
# Footer
# ==============================
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; background: linear-gradient(45deg, #f8f9fa, #e9ecef); border-radius: 15px;'>
    <h3 style='color: #2E86AB;'>Built with ❤️ using Streamlit & TensorFlow</h3>
    <p style='color: #666;'>Advanced Facial Emotion Recognition AI | Now with Automatic Face Detection</p>
</div>
""", unsafe_allow_html=True)

# Add requirements installation note
with st.expander("📦 Installation Requirements"):
    st.code("""
pip install streamlit tensorflow mtcnn pillow opencv-python numpy matplotlib plotly streamlit-lottie requests
""", language="bash")
    st.info("Make sure to install MTCNN for face detection functionality")
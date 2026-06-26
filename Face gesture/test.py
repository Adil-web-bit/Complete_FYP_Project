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
</style>
""", unsafe_allow_html=True)


# Load Lottie animation
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_ai = load_lottie_url("https://assets1.lottiefiles.com/packages/lf20_5tkzkblw.json")


# Load the trained model
@st.cache_resource
def load_emotion_model():
    return load_model('emotion_detection_model.keras')


model = load_emotion_model()

# Emotion labels and colors
emotion_labels = ['Angry 😠', 'Disgust 🤢', 'Fear 😨', 'Happy 😊', 'Neutral 😐', 'Sad 😢', 'Surprise 😲']
emotion_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFE66D', '#95A5A6', '#5C6BC0', '#FFA726']
emotion_emojis = ['😠', '🤢', '😨', '😊', '😐', '😢', '😲']

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
        The model can identify <strong>7 different emotions</strong> using deep learning.
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
    <p><strong>Dataset:</strong> FER-2013</p>
    <p><strong>Classes:</strong> 7 Emotions</p>
    <p><strong>Architecture:</strong> CNN</p>
    <p><strong>Accuracy:</strong> 82%</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 💡 Pro Tips")
    tips = [
        "✅ Use clear frontal face images",
        "✅ Ensure good lighting conditions",
        "✅ Neutral background works best",
        "❌ Avoid blurry or side profiles",
        "❌ Don't use heavy filters",
        "❌ Single face per image"
    ]

    for tip in tips:
        st.markdown(f"- {tip}")

# ==============================
# Main content area
# ==============================
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="sub-header">📤 Upload Your Image</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="upload-area">
        <h3 style='color: #2E86AB;'>Drag & Drop or Click to Upload</h3>
        <p style='color: #666;'>Supported formats: JPG, JPEG, PNG</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        " ",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of a face",
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        # ----- Minimum size handling -----
        min_required = 250
        min_allowed = 30

        w, h = image.size
        min_dim = min(w, h)

        if image.width < min_allowed or image.height < min_allowed:
            st.error(f"❌ Image is too small! Minimum acceptable size is {min_allowed}px.")
            st.stop()

        # Resize logic
        if min_dim < min_required:
            scale = min_required / min_dim  # scale to reach 250px min
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = image.resize((new_w, new_h))
        # ----------------------------------


        st.markdown("### 📷 Your Uploaded Image")

        # Convert to base64 for HTML display
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode()

        # Fixed preview size (max 200x200)
        st.markdown(f"""
        <div style='text-align: center;'>
            <img src='data:image/png;base64,{img_str}'
                 style='max-width:400px; max-height:400px;
                        width:auto; height:auto;
                        border-radius:10px; object-fit:contain; 
                        box-shadow:0 4px 10px rgba(0,0,0,0.1);'>
            <p style='color:#555;'>Ready for analysis!</p>
        </div>
        """, unsafe_allow_html=True)


        # Preprocess function
        def preprocess_image(image):
            image = image.convert('L')
            image = image.resize((48, 48))
            image_array = np.array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)
            image_array = np.expand_dims(image_array, axis=-1)
            return image_array

with col2:
    st.markdown('<div class="sub-header">🔍 Analysis Results</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        with st.spinner('🤖 AI is analyzing emotions...'):
            progress_bar = st.progress(0)
            for i in range(100):
                pass
            progress_bar.progress(100)

            processed_image = preprocess_image(image)
            predictions = model.predict(processed_image, verbose=0)

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
# Features section
# ==============================
st.markdown("---")
st.markdown('<div class="sub-header" style="text-align: center;">🚀 Why Choose Our AI?</div>', unsafe_allow_html=True)

col3, col4, col5 = st.columns(3)

with col3:
    st.markdown("""
    <div class="feature-card">
        <h3>🎯 High Accuracy</h3>
        <p>Trained on thousands of facial images for precise emotion detection</p>
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
        <h3>🔍 7 Emotions</h3>
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
        <h3>🎨 User-Friendly</h3>
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
# Tech stack
# ==============================
st.markdown("---")
st.markdown('<div class="sub-header" style="text-align: center;">🛠️ Powered By</div>', unsafe_allow_html=True)

tech_cols = st.columns(5)
tech_icons = ["🤖", "📚", "🎯", "⚡", "🔍"]
tech_names = ["TensorFlow", "Keras", "CNN", "Streamlit", "OpenCV"]

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
    <p style='color: #666;'>Advanced Facial Emotion Recognition AI | Making AI Accessible to Everyone</p>
</div>
""", unsafe_allow_html=True)

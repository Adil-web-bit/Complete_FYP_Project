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
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
        border: 2px solid #667eea;
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        transition: all 0.3s ease;
        height: 100%;
        cursor: pointer;
    }
    
    .feature-card:hover {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-color: #667eea;
        box-shadow: 0 15px 30px rgba(102, 126, 234, 0.3);
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
        color: #667eea;
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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.6rem 1.3rem;
        border-radius: 25px;
        font-size: 0.95rem;
        font-weight: 600;
        margin: 0.4rem;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero-section">
    <h1 class="main-title">🎭 Human Behaviour Predictor</h1>
    <p class="subtitle">Advanced AI-Powered Emotion Recognition System</p>
    <p style="font-size: 1.1rem; opacity: 0.9;">Analyzing human emotions through facial expressions, voice patterns, and typing behavior using AI</p>
    <p class="university">🎓 University of Engineering and Technology, Taxila | Final Year Project 2025-2026</p>
</div>
""", unsafe_allow_html=True)

# Content Section
st.markdown('<div class="content-section">', unsafe_allow_html=True)

# Project Overview
st.markdown("<h2 style='text-align: center; color: #2d3748; font-weight: 700; margin-bottom: 3rem; font-size: 2.5rem;'>🚀 What We Offer</h2>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="large")

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

with col3:
    st.markdown("""
    <div class="feature-card">
        <span class="feature-icon">⌨️</span>
        <h3 class="feature-title">Typing Analysis</h3>
        <p class="feature-desc">
            Behavioural analysis that captures typing rhythm, key timing,
            pauses, and interaction patterns to estimate emotional state through typing dynamics.
        </p>
        <div style="margin-top: 2rem;">
            <div class="tech-badge">Keystroke</div>
            <div class="tech-badge">Typing Rhythm</div>
            <div class="tech-badge">Random Forest</div>
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
        <div class="stat-number">3</div>
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
    <p style='font-size: 1rem; opacity: 0.85;'>👈 Select Facial Expression Analysis, Voice Analysis, or Typing Analysis from the navigation menu</p>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

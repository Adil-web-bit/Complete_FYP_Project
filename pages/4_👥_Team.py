"""
Team Page
"""

import streamlit as st

# Page config
st.set_page_config(
    page_title="Our Team",
    page_icon="👥",
    layout="wide"
)

# Professional CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(to bottom, #f8f9fa 0%, #ffffff 100%);
    }
    
    .page-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin: 1rem 0;
    }
    
    .subtitle {
        text-align: center;
        color: #718096;
        font-size: 1.1rem;
        margin-bottom: 3rem;
    }
    
    .team-card {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        text-align: center;
        transition: all 0.3s ease;
        border: 1px solid #e2e8f0;
        height: 100%;
    }
    
    .team-card:hover {
        box-shadow: 0 12px 24px rgba(102, 126, 234, 0.15);
        transform: translateY(-5px);
    }
    
    .team-avatar {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        margin: 0 auto 1.5rem auto;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 3rem;
        color: white;
    }
    
    .team-name {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2d3748;
        margin-bottom: 0.5rem;
    }
    
    .team-role {
        font-size: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .team-bio {
        color: #718096;
        line-height: 1.6;
        font-size: 0.95rem;
    }
    
    .project-info {
        background: white;
        padding: 2.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        margin: 2rem 0;
        border-left: 5px solid #667eea;
    }
    
    .tech-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1.2rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 600;
        margin: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# Main content
st.markdown('<h1 class="page-title">👥 Our Team</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Meet the team behind Human Behaviour Predictor</p>', unsafe_allow_html=True)

# Project Overview
st.markdown("""
<div class="project-info">
    <h2 style='color: #2d3748; font-weight: 700; margin-bottom: 1rem;'>📚 Project Overview</h2>
    <p style='color: #4a5568; line-height: 1.8; font-size: 1.05rem;'>
        <strong>Human Behaviour Predictor</strong> is a cutting-edge AI system that combines computer vision and 
        audio processing to analyze and predict human emotions. Using deep learning models, the system can 
        detect emotions from both facial expressions and voice patterns, providing comprehensive behavioral insights.
    </p>
    <div style='margin-top: 1.5rem;'>
        <span class='tech-badge'>Deep Learning</span>
        <span class='tech-badge'>Computer Vision</span>
        <span class='tech-badge'>Audio Processing</span>
        <span class='tech-badge'>TensorFlow</span>
        <span class='tech-badge'>Python</span>
        <span class='tech-badge'>Streamlit</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Team Members
st.markdown("<h2 style='text-align: center; color: #2d3748; font-weight: 700; margin: 3rem 0 2rem 0;'>🌟 Team Members</h2>", unsafe_allow_html=True)

# Add your team members here
team_members = [
    {
        "name": "Muhammad Dilawar Akram",
        "role": "AI Automation & SaaS Development",
        "emoji": "👨‍💻",
        "bio": "Final year Computer Science student with expertise in AI Automation and AI-driven automation. Developed LeadLoop AI, a comprehensive SaaS product that automates the entire email marketing process."
    },
    {
        "name": "Muhammad Adil",
        "role": "Web Development & AI Specialist",
        "emoji": "👨‍💻",
        "bio": "Final-year Computer Science undergraduate at UET Taxila with strong expertise in web development and artificial intelligence. Works as a dedicated freelancer on Fiverr, handling diverse projects independently."
    },
    {
        "name": "Muhammad Ahmad",
        "role": "Machine Learning & NLP Expert",
        "emoji": "👨‍💻",
        "bio": "Final-year Bachelor's student in Computer Science at UET Taxila, specializing in Machine Learning and Natural Language Processing. Coursera-certified in Advanced Data Science with proven expertise in AI systems."
    }
]

cols = st.columns(3, gap="large")

for idx, member in enumerate(team_members):
    with cols[idx % 3]:
        st.markdown(f"""
        <div class="team-card">
            <div class="team-avatar">{member['emoji']}</div>
            <div class="team-name">{member['name']}</div>
            <div class="team-role">{member['role']}</div>
            <div class="team-bio">{member['bio']}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

# Supervisor Section
st.markdown("<hr style='margin: 3rem 0; border: none; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: #2d3748; font-weight: 700; margin: 2rem 0;'>�‍🏫 Project Supervisor</h2>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("""
    <div class="team-card">
        <div class="team-avatar">👩‍🏫</div>
        <div class="team-name">Miss Asima Ismail</div>
        <div class="team-role">Project Supervisor</div>
        <div class="team-bio">
            Supervisor at University of Engineering and Technology Taxila<br>
            Guiding expertise in Artificial Intelligence and Machine Learning
        </div>
    </div>
    """, unsafe_allow_html=True)

# University Info
st.markdown("<hr style='margin: 3rem 0; border: none; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)

st.markdown("""
<div class="project-info" style='text-align: center; border-left: none;'>
    <h2 style='color: #2d3748; font-weight: 700; margin-bottom: 1rem;'>🎓 University</h2>
    <p style='color: #4a5568; font-size: 1.1rem; font-weight: 600;'>
        University of Engineering and Technology, Taxila
    </p>
    <p style='color: #718096; margin-top: 0.5rem;'>Final Year Project 2025-2026</p>
</div>
""", unsafe_allow_html=True)

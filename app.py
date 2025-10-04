import streamlit as st
import joblib
from preprocess import clean_tweet
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Cyberbullying Analyzer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #000000;
    }
    [data-testid="stSidebar"] {
        background-color: #1a1a1a;
    }
    [data-testid="stTextArea"] textarea {
        background-color: #2a2a2a;
        color: #ffffff;
    }
    .stTextInput input {
        background-color: #2a2a2a;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FFF3E0;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #FB8C00;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #43A047;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #FFEBEE;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #E53935;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-size: 1.1rem;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1565C0;
    }
    </style>
""", unsafe_allow_html=True)

# Load saved artifacts
@st.cache_resource
def load_models():
    model = joblib.load("cyberbullying_lr_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, vectorizer, label_encoder

model, vectorizer, label_encoder = load_models()

# Prediction function
def predict_cyberbullying(text):
    import re
    from nltk.corpus import stopwords

    # Clean
    text = clean_tweet(text)

    # Tokenize & remove stopwords
    tokens = text.split()
    stop_words = set(stopwords.words("english"))
    tokens = [w for w in tokens if w not in stop_words]

    final_text = " ".join(tokens)

    # Vectorize & predict
    X_vec = vectorizer.transform([final_text])
    pred_num = model.predict(X_vec)[0]
    pred_label = label_encoder.inverse_transform([pred_num])[0]
    proba = model.predict_proba(X_vec)[0]
    confidence = max(proba) * 100
    
    return pred_label, confidence

# Category information
category_info = {
    "religion": {
        "icon": "🕉️",
        "color": "#E53935",
        "description": "Religious-based harassment or discrimination",
        "examples": "Attacks based on religious beliefs, practices, or affiliations"
    },
    "age": {
        "icon": "👶",
        "color": "#FB8C00",
        "description": "Age-based discrimination or bullying",
        "examples": "Harassment targeting someone's age, whether young or old"
    },
    "ethnicity": {
        "icon": "🌍",
        "color": "#8E24AA",
        "description": "Ethnic or racial harassment",
        "examples": "Discrimination based on race, nationality, or ethnic background"
    },
    "gender": {
        "icon": "⚧️",
        "color": "#D81B60",
        "description": "Gender-based harassment or sexism",
        "examples": "Attacks based on gender identity, expression, or stereotypes"
    },
    "not_cyberbullying": {
        "icon": "✅",
        "color": "#43A047",
        "description": "No cyberbullying detected",
        "examples": "Normal, respectful communication without harmful intent"
    },
    "other_cyberbullying": {
        "icon": "⚠️",
        "color": "#F4511E",
        "description": "General harassment or bullying",
        "examples": "Bullying that doesn't fit specific categories but is still harmful"
    }
}

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/security-shield-green.png", width=80)
    st.title("About This Tool")
    
    st.markdown("""
    This tool analyzes text messages, tweets, and comments to identify potential cyberbullying content. It categorizes harmful language into specific types, helping users recognize and address online harassment.
    
    **🎯 Key Features:**
    - Real-time text analysis
    - 6 category classification
    - 82.4% accuracy rate
    - Trained on 47,000+ tweets
    
    **📊 Model Details:**
    - Algorithm: Logistic Regression
    - Features: TF-IDF (5,000 features)
    - Training Data: Twitter dataset
    """)
    
    st.markdown("---")
    st.title("Categories Detected")
    for cat, info in category_info.items():
        st.markdown(f"{info['icon']} {cat.replace('_', ' ').title()}")
    
    st.markdown("---")
    st.info("💡 **Tip:** This tool helps identify potentially harmful content but should be used alongside human judgment.")

# Main content
st.markdown("<h2 style='text-align: center; color: #1E88E5;'>🛡️ Cyberbullying Analyzer</h2>", unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Text Analysis for Online Safety</p>', unsafe_allow_html=True)

# Introduction section
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Model Accuracy", "82.4%", help="Overall accuracy on test dataset")
with col2:
    st.metric("Training Samples", "47,656", help="Tweets used for training")
with col3:
    st.metric("Categories", "6", help="Types of cyberbullying detected")

st.markdown("---")

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["🔍 Analyze Text", "📖 Understanding Cyberbullying", "📊 About the Model"])

with tab1:
    st.markdown("### Enter Text for Analysis")
    st.markdown("Type or paste any text below to check if it contains cyberbullying content.")
    
    user_input = st.text_area(
        "Text Input:",
        placeholder="Example: 'You're such a loser and nobody likes you.'",
        height=150,
        help="Enter any text message, tweet, or comment you want to analyze"
    )
    
    analyze_button = st.button("🔍 Analyze Text", type="primary")
    
    if analyze_button:
        if user_input.strip() != "":
            with st.spinner("Analyzing text..."):
                prediction, confidence = predict_cyberbullying(user_input)
                
                st.markdown("---")
                st.markdown("### 📋 Analysis Results")
                
                st.markdown("**Original Text:**")
                st.info(user_input)
                
                info = category_info[prediction]
                
                st.markdown("**Classification:**")
                
                if prediction == "not_cyberbullying":
                    st.success(f"{info['icon']} **{prediction.replace('_', ' ').title()}** (Confidence: {confidence:.1f}%)")
                    st.markdown(f"**Description:** {info['description']}")
                else:
                    st.error(f"{info['icon']} **{prediction.replace('_', ' ').title()}** (Confidence: {confidence:.1f}%)")
                    st.markdown(f"**Description:** {info['description']}")
                
                # Additional information
                st.markdown("---")
                st.markdown("### 💡 What This Means")
                
                if prediction == "not_cyberbullying":
                    st.success("✅ This text appears to be safe and does not contain obvious cyberbullying content.")
                else:
                    st.error(f"⚠️ This text has been classified as **{prediction.replace('_', ' ')}** cyberbullying.")
                    st.warning("""
                    **Recommended Actions:**
                    - Report this content to platform moderators
                    - Block or mute the sender if on social media
                    - Document the incident if it's part of ongoing harassment
                    - Seek support from trusted friends, family, or professionals
                    """)
        else:
            st.warning("⚠️ Please enter some text to analyze.")

with tab2:
    st.markdown("### 🧠 Understanding Cyberbullying")
    
    st.markdown("""
    Cyberbullying is the use of digital communication tools to harass, threaten, embarrass, or target another person. 
    It can have serious emotional and psychological impacts on victims.
    """)
    
    st.markdown("---")
    st.markdown("### 📱 Types of Cyberbullying Detected by This Tool")
    
    for cat, info in category_info.items():
        if cat != "not_cyberbullying":
            with st.expander(f"{info['icon']} {cat.replace('_', ' ').title()} Cyberbullying"):
                st.markdown(f"**Description:** {info['description']}")
                st.markdown(f"**Examples:** {info['examples']}")
                st.markdown("**Impact:** This type of harassment can cause emotional distress, anxiety, and feelings of isolation.")
    
    st.markdown("---")
    st.markdown("### 🆘 Resources for Help")
    
    st.markdown("""
    **🌐 Online Resources:**
    - **CyberBullying Research Center** - Research and resources on cyberbullying
    - **StopBullying.gov** - Government resource for bullying prevention
    - **National Bullying Prevention Center** - Educational materials and support
    - **Crisis Text Line** - Text-based crisis support available globally
    - **International Association for Suicide Prevention** - Global mental health resources
    - **Childline International** - Child protection helplines worldwide
    
    💡 *Search for local support services in your country or region for immediate help.*
    """)

with tab3:
    st.markdown("### 📊 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🤖 Technical Details")
        st.markdown("""
        **Algorithm:** Logistic Regression  
        **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency)  
        **Features:** 5,000 n-grams (unigrams and bigrams)  
        **Training Size:** 38,124 samples  
        **Test Size:** 9,532 samples  
        **Overall Accuracy:** 82.4%
        """)
        
        st.markdown("#### 🔄 Preprocessing Steps")
        st.markdown("""
        1. Convert text to lowercase
        2. Remove URLs and mentions
        3. Remove special characters
        4. Remove stop words
        5. Tokenization
        6. TF-IDF vectorization
        """)
    
    with col2:
        st.markdown("#### 📈 Performance Metrics")
        
        # Performance data
        performance_data = {
            "Category": ["Age", "Ethnicity", "Gender", "Not Cyberbullying", "Other", "Religion"],
            "Precision": [0.95, 0.97, 0.92, 0.62, 0.57, 0.95],
            "Recall": [0.98, 0.97, 0.81, 0.50, 0.73, 0.95],
            "F1-Score": [0.96, 0.97, 0.86, 0.56, 0.64, 0.95]
        }
        
        df_performance = pd.DataFrame(performance_data)
        st.dataframe(df_performance, hide_index=True, use_container_width=True)
        
        st.markdown("#### ⚖️ Model Strengths")
        st.success("""
        ✅ Excellent at detecting religion, age, and ethnicity-based cyberbullying  
        ✅ High precision across most categories  
        ✅ Fast inference time for real-time analysis
        """)
        
        st.markdown("#### ⚠️ Limitations")
        st.warning("""
        - Lower accuracy on general cyberbullying and non-cyberbullying content
        - May struggle with sarcasm or context-dependent language
        - Limited to English language text
        - Trained on Twitter data, may not generalize to all platforms
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🛡️ <strong>Cyberbullying Analyzer</strong> | Built with Machine Learning & Streamlit</p>
    <p style='font-size: 0.9rem;'>This tool is designed to assist in identifying potentially harmful content. 
    Always use human judgment and consider context when making decisions about online content.</p>
    <p style='font-size: 0.8rem; margin-top: 1rem;'>⚠️ If you or someone you know is experiencing cyberbullying, 
    please reach out to a trusted adult, counselor, or contact appropriate support services.</p>
</div>
""", unsafe_allow_html=True)
import streamlit as st
import pickle
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import google.generativeai as genai
from pathlib import Path
from dotenv import load_dotenv
import os
from google.api_core.exceptions import ResourceExhausted

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------
load_dotenv()

# Try Streamlit Cloud secrets first, then local .env
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    API_KEY = os.getenv("GEMINI_API_KEY")  # ✅ FIXED

if not API_KEY:
    st.error("Gemini API key not found.")
    st.stop()

# Configure Gemini
genai.configure(api_key=API_KEY)

# --------------------------------------------------
# Initialize Gemini model ONCE
# --------------------------------------------------
gemini_model = genai.GenerativeModel("models/gemini-2.5-flash")  # ✅ FIXED

# --------------------------------------------------
# Download stopwords
# --------------------------------------------------
nltk.download("stopwords")

# --------------------------------------------------
# Text cleaning function
# --------------------------------------------------
def clean_caption(text):
    stop_words = set(stopwords.words("english"))

    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = []
    for word in text.split():
        if word not in stop_words:
            words.append(word)

    return " ".join(words)

# --------------------------------------------------
# Gemini suggestion function
# --------------------------------------------------
def get_gemini_suggestions(caption, score):
    try:
        prompt = f"""
Caption: {caption}
Predicted Engagement Score: {round(score, 2)}

Give:
- Short explanation
- 3 improvement tips
- Improved caption
- 5 hashtags
"""

        response = gemini_model.generate_content(
            prompt,
            request_options={"retry": None}
        )

        # ✅ HANDLE ALL CASES
        if hasattr(response, "text") and response.text:
            return response.text.strip()

        if hasattr(response, "candidates") and response.candidates:
            try:
                return response.candidates[0].content.parts[0].text.strip()
            except:
                pass

        return "⚠️ AI could not generate suggestions right now. Try again later."

    except ResourceExhausted as e:
        return f"⚠️ Quota exceeded: {e}"
    except Exception as e:
        return f"⚠️ Error: {e}"

# --------------------------------------------------
# Load trained ML model
# --------------------------------------------------
BASE_DIR = Path(__file__).parent
MODEL_FILE = BASE_DIR / "ads_predictor.pkl"

try:
    with open(MODEL_FILE, "rb") as file:
        prediction_model = pickle.load(file)
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# --------------------------------------------------
# Streamlit page settings
# --------------------------------------------------
st.set_page_config(
    page_title="Ad Performance Predictor",
    layout="centered"
)

st.title("📊 Ad Performance Predictor")
st.write("Predict ad engagement and get simple AI suggestions.")

# --------------------------------------------------
# Session state
# --------------------------------------------------
if "ai_result" not in st.session_state:
    st.session_state.ai_result = None

if "predicted_score" not in st.session_state:
    st.session_state.predicted_score = None

# --------------------------------------------------
# User inputs
# --------------------------------------------------
caption = st.text_area(
    "Ad Caption",
    placeholder="Write your ad caption here..."
)

account_name = st.text_input(
    "Brand / Account Name",
    placeholder="Example: Nike"
)

platform = st.selectbox(
    "Platform",
    ["Facebook", "Instagram", "Twitter", "LinkedIn"]
)

comment_count = st.number_input(
    "Expected Comments",
    min_value=0,
    step=1
)

like_count = st.number_input(
    "Expected Likes",
    min_value=0,
    step=1
)

sentiment_score = st.slider(
    "Sentiment Score",
    -1.0,
    1.0,
    0.0
)

# --------------------------------------------------
# Predict button
# --------------------------------------------------
if st.button("Predict Engagement", key="predict_btn"):

    cleaned_text = clean_caption(caption)

    data = pd.DataFrame([{
        "caption": cleaned_text,
        "account_name": account_name,
        "platform": platform,
        "comment_count": comment_count,
        "like_count": like_count,
        "caption_length": len(caption),
        "word_count": len(caption.split()),
        "sentiment_score": sentiment_score
    }])

    predicted_score = prediction_model.predict(data)[0]

    st.session_state.predicted_score = predicted_score
    st.session_state.ai_result = None

# --------------------------------------------------
# Show prediction
# --------------------------------------------------
if st.session_state.predicted_score is not None:
    st.success(f"Predicted Engagement Score: {round(st.session_state.predicted_score, 2)}")

# --------------------------------------------------
# Generate AI Suggestions
# --------------------------------------------------
if st.session_state.predicted_score is not None:

    if st.button("Generate AI Suggestions", key="ai_btn"):
        with st.spinner("Getting AI suggestions..."):
            st.session_state.ai_result = get_gemini_suggestions(
                caption,
                st.session_state.predicted_score
            )

# --------------------------------------------------
# Show AI output (FIXED CONDITION)
# --------------------------------------------------
if st.session_state.ai_result is not None:
    st.subheader("🤖 AI Suggestions")
    st.write(st.session_state.ai_result)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("Built with Streamlit + Gemini")

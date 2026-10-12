import os
import pickle
import streamlit as st
import helper

# Page configuration
st.set_page_config(
    page_title="Quora Duplicate Question Detector",
    page_icon="❓",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        margin-bottom: 25px;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #b92b27;
        margin-bottom: 8px;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #555555;
        margin-bottom: 20px;
    }
    .result-card-dup {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 2px solid #28a745;
        border-radius: 12px;
        padding: 22px;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .result-card-not-dup {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 2px solid #dc3545;
        border-radius: 12px;
        padding: 22px;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .result-title {
        font-size: 1.7rem;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .result-desc {
        font-size: 1.05rem;
        color: #333333;
    }
    .stTextArea textarea {
        font-size: 1rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    """Load the trained model and CountVectorizer objects."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "model.pkl")
    cv_path = os.path.join(current_dir, "cv.pkl")

    if not os.path.exists(model_path) or not os.path.exists(cv_path):
        raise FileNotFoundError("Could not find 'model.pkl' or 'cv.pkl' in the project directory.")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(cv_path, "rb") as f:
        cv = pickle.load(f)

    return model, cv


# Sample test pairs for user convenience
SAMPLE_PAIRS = {
    "Custom (Type your own)": ("", ""),
    "Duplicate Example 1 (Capital of India)": (
        "Where is the capital of India?",
        "What is the current capital of India?"
    ),
    "Duplicate Example 2 (Learning Machine Learning)": (
        "How can I learn machine learning?",
        "What is the best way to learn machine learning from scratch?"
    ),
    "Duplicate Example 3 (Weight Loss)": (
        "How can I lose weight quickly?",
        "What are the best tips to lose weight fast?"
    ),
    "Not Duplicate Example 1 (Baking vs Car)": (
        "How do I bake a chocolate cake?",
        "Where can I buy a used sports car?"
    ),
    "Not Duplicate Example 2 (Apple Fruit vs Apple Inc)": (
        "What are the nutritional health benefits of eating an apple daily?",
        "What is the stock market valuation and financial growth of Apple Inc?"
    )
}

# Load artifacts
try:
    model, cv = load_artifacts()
    artifacts_loaded = True
except Exception as e:
    artifacts_loaded = False
    load_error_msg = str(e)

# Sidebar with model metadata and instructions
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/9/91/Quora_logo_2015.svg", width=180)
    st.markdown("### Model Overview")
    st.markdown(
        """
        This application uses NLP feature engineering and machine learning to detect duplicate questions from the **Quora Question Pairs** dataset.
        
        **Pipeline details:**
        - **Model:** Random Forest Classifier
        - **Handcrafted Features (22):**
          - Length & Word counts
          - Common words & word share
          - Token-level ratios (stops & non-stops)
          - Fuzzy string matching (Rapidfuzz)
          - Longest common substring
        - **BoW Features (6,000):**
          - CountVectorizer (3,000 per question)
        - **Total Features:** 6,022
        """
    )

    if artifacts_loaded:
        st.success("Model & Vectorizer Loaded")
    else:
        st.error("Failed to load artifacts")

# Main Page Header
st.markdown("""
<div class="main-header">
    <div class="main-title">Quora Question Pairs Duplicate Detector</div>
    <div class="subtitle">Enter two questions below to predict whether they have identical intent.</div>
</div>
""", unsafe_allow_html=True)

if not artifacts_loaded:
    st.error(f"Error loading model artifacts: {load_error_msg}")
    st.stop()

# Quick sample selector
selected_sample = st.selectbox(
    "Choose a preset example or enter custom questions:",
    options=list(SAMPLE_PAIRS.keys()),
    index=0
)

default_q1, default_q2 = SAMPLE_PAIRS[selected_sample]

# Question Inputs
col1, col2 = st.columns(2)

with col1:
    q1_input = st.text_area(
        "Question 1",
        value=default_q1,
        placeholder="e.g. How can I improve my English speaking skills?",
        height=130,
        key="question_1"
    )

with col2:
    q2_input = st.text_area(
        "Question 2",
        value=default_q2,
        placeholder="e.g. What are the best ways to speak English fluently?",
        height=130,
        key="question_2"
    )

# Action button
predict_clicked = st.button("Detect Duplicate", type="primary", use_container_width=True)

if predict_clicked:
    q1 = q1_input.strip()
    q2 = q2_input.strip()

    if not q1 or not q2:
        st.warning("Please enter both questions to run the prediction.")
    else:
        with st.spinner("Analyzing semantic similarity and calculating features..."):
            # Compute feature vector
            feat_vector, details = helper.query_point_creator(q1, q2, cv)

            # Predict
            pred = model.predict(feat_vector)[0]
            probs = model.predict_proba(feat_vector)[0]
            prob_not_dup = float(probs[0])
            prob_dup = float(probs[1])

        # Display result banner
        if pred == 1:
            st.markdown(f"""
            <div class="result-card-dup">
                <div class="result-title" style="color: #155724;">Duplicate Questions</div>
                <div class="result-desc">Both questions appear to share the same underlying intent.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card-not-dup">
                <div class="result-title" style="color: #721c24;">Not Duplicate Questions</div>
                <div class="result-desc">These questions have different meanings or ask for different information.</div>
            </div>
            """, unsafe_allow_html=True)

        # Confidence Metrics
        st.markdown("### Confidence Breakdown")
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric(
                label="Duplicate Probability",
                value=f"{prob_dup * 100:.1f}%",
                delta=f"{'+' if pred == 1 else ''}{(prob_dup - 0.5) * 100:.1f}% vs threshold"
            )
            st.progress(prob_dup)

        with m_col2:
            st.metric(
                label="Not Duplicate Probability",
                value=f"{prob_not_dup * 100:.1f}%",
                delta=f"{'+' if pred == 0 else ''}{(prob_not_dup - 0.5) * 100:.1f}% vs threshold"
            )
            st.progress(prob_not_dup)

        # Extracted NLP Feature Details
        with st.expander("View Extracted Feature Insights", expanded=False):
            st.markdown("#### Preprocessed Text")
            st.code(f"Q1: {details['q1_preprocessed']}\nQ2: {details['q2_preprocessed']}")

            st.markdown("#### Word & Token Metrics")
            t_col1, t_col2, t_col3 = st.columns(3)
            with t_col1:
                st.metric("Common Words", details["common_words"])
            with t_col2:
                st.metric("Total Words", details["total_words"])
            with t_col3:
                st.metric("Word Share", f"{details['word_share'] * 100:.1f}%")

            st.markdown("#### Fuzzy Match Scores (0 - 100)")
            f_col1, f_col2, f_col3, f_col4 = st.columns(4)
            with f_col1:
                st.metric("Fuzz QRatio", f"{details['fuzz_ratio']:.1f}")
            with f_col2:
                st.metric("Partial Ratio", f"{details['fuzz_partial_ratio']:.1f}")
            with f_col3:
                st.metric("Token Sort Ratio", f"{details['token_sort_ratio']:.1f}")
            with f_col4:
                st.metric("Token Set Ratio", f"{details['token_set_ratio']:.1f}")

            st.markdown("#### Length Metrics")
            l_col1, l_col2 = st.columns(2)
            with l_col1:
                st.metric("Absolute Token Diff", details["abs_len_diff"])
            with l_col2:
                st.metric("Longest Substr Ratio", f"{details['longest_substr_ratio']:.2f}")

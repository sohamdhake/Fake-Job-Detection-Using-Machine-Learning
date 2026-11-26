"""
Fake Job Detection System - Streamlit Web Application
=====================================================
A college AIML mini-project web interface to detect fraudulent job postings
using TF-IDF feature extraction and Logistic Regression.
"""

import os
import re
import json
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set Page Config
st.set_page_config(
    page_title="Fake Job Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "model.joblib")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")


# Custom CSS for polished College Project UI
st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        color: white !important;
        font-size: 2.2rem;
        margin-bottom: 6px;
        font-weight: 700;
    }
    .main-header p {
        color: #e0f2fe !important;
        font-size: 1.05rem;
        margin: 0;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
    }
    .alert-fake {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border-left: 6px solid #ef4444;
        padding: 20px;
        border-radius: 8px;
        margin-top: 20px;
        color: #991b1b;
    }
    .alert-real {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-left: 6px solid #22c55e;
        padding: 20px;
        border-radius: 8px;
        margin-top: 20px;
        color: #166534;
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .badge-red {
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #fca5a5;
    }
    .badge-green {
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #86efac;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def clean_text(text: str) -> str:
    """Cleans input text identical to training pipeline."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource
def load_models():
    """Loads the trained ML model and TF-IDF vectorizer."""
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            vectorizer = joblib.load(VECTORIZER_PATH)
            return model, vectorizer, True
        except Exception as e:
            st.error(f"Error loading model files: {e}")
            return None, None, False
    return None, None, False


def load_metrics():
    """Loads saved model evaluation metrics."""
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


# Suspicious keywords dictionary for heuristic risk breakdown
SUSPICIOUS_KEYWORDS = [
    "wire transfer",
    "cashapp",
    "venmo",
    "gift card",
    "registration fee",
    "starter kit",
    "no experience required",
    "earn daily",
    "unlimited earning",
    "immediate hire",
    "interview via telegram",
    "interview via whatsapp",
    "processing fee",
    "personal assistant",
    "repackaging",
    "mystery shopper",
    "crypto",
    "bitcoin",
    "western union",
]


def detect_suspicious_patterns(text: str):
    """Finds common scam phrases in text for explainability."""
    text_lower = text.lower()
    found = []
    for kw in SUSPICIOUS_KEYWORDS:
        if kw in text_lower:
            found.append(kw)
    return found


# Sample presets for quick testing and college viva demonstrations
PRESETS = {
    "-- Select a Sample Preset --": {
        "title": "",
        "company": "",
        "desc": "",
    },
    "🚨 Sample 1: High-Risk Fake Job (Data Entry / CashApp Scam)": {
        "title": "Urgent Work From Home Data Entry Clerk",
        "company": "Apex Global Solutions LLC",
        "desc": (
            "We are urgently looking for remote data entry assistants! Work 1 to 2 hours per day from home and earn $500 to $800 daily. "
            "No prior experience, resume, or educational degree needed. Instant hiring without formal interview. "
            "You will receive daily payments directly. To receive your company-provided laptop and training software kit, "
            "a refundable security deposit of $75 must be transferred via CashApp, Venmo, or Bitcoin to our distribution agent."
        ),
    },
    "🚨 Sample 2: High-Risk Fake Job (Executive Assistant / Check Scam)": {
        "title": "Virtual Personal & Executive Assistant",
        "company": "Global Trade Ventures",
        "desc": (
            "Seeking an enthusiastic personal assistant to handle correspondence and light purchases. Flexible hours, $45/hour. "
            "Interview conducted solely via Telegram or WhatsApp. We will mail you a company cashier's check to purchase office supplies; "
            "you must wire the remaining balance immediately via Western Union back to our supplier."
        ),
    },
    "✅ Sample 3: Legitimate Job (Senior Python Backend Engineer)": {
        "title": "Senior Python Backend Engineer",
        "company": "FinTech Innovations Corp",
        "desc": (
            "FinTech Innovations Corp is seeking a Senior Python Engineer with 4+ years of hands-on experience building distributed backend systems. "
            "Required skills: Python 3, FastAPI or Django, PostgreSQL, Docker, and AWS. "
            "Key responsibilities include architecting high-throughput REST APIs, writing comprehensive unit tests, and participating in code reviews. "
            "We offer competitive salary ($120k-$150k), comprehensive medical/dental insurance, 401(k) matching, and 20 days paid vacation."
        ),
    },
    "✅ Sample 4: Legitimate Job (Marketing & Communications Specialist)": {
        "title": "Marketing & Communications Specialist",
        "company": "NextGen Media Group",
        "desc": (
            "NextGen Media Group is looking for an organized Marketing Specialist to lead our digital campaigns. "
            "Bachelor's degree in Marketing, PR, or related field required. Must have 2+ years experience with Google Analytics, SEO, and social media scheduling. "
            "Responsibilities include drafting press releases, monitoring campaign metrics, and collaborating with our design team. "
            "Standard background check and professional reference verification will be conducted prior to onboarding."
        ),
    },
}


# --- SIDEBAR ---
with st.sidebar:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/2910/2910791.png",
        width=70,
    )
    st.title("Project Details")
    st.caption("College AIML Mini Project")

    st.markdown("---")
    st.markdown("### ⚙️ Model Pipeline")
    st.markdown("- **Algorithm:** Logistic Regression")
    st.markdown("- **Feature Extraction:** TF-IDF (1-2 N-grams)")
    st.markdown("- **Class Balancing:** Balanced Weights")
    st.markdown("- **Dataset:** EMSCAD / Kaggle Real or Fake Job Postings (~18k records)")

    st.markdown("---")
    st.markdown("### 📁 Model Status")
    model, vectorizer, is_model_loaded = load_models()
    if is_model_loaded:
        st.success(" Model loaded and ready!")
    else:
        st.error("❌ Model not found in `models/`")
        st.info("Run `py train.py` in your terminal to train and save the model.")

    metrics = load_metrics()
    if metrics:
        st.markdown("---")
        st.markdown("### 🎯 Stored Test Metrics")
        st.markdown(f"- **Accuracy:** `{metrics['accuracy']*100:.2f}%`")
        st.markdown(f"- **Precision:** `{metrics['precision']*100:.2f}%`")
        st.markdown(f"- **Recall:** `{metrics['recall']*100:.2f}%`")
        st.markdown(f"- **F1-Score:** `{metrics['f1_score']*100:.2f}%`")


# --- MAIN HEADER ---
st.markdown(
    """
    <div class="main-header">
        <h1>🛡️ Fake Job Detection System</h1>
        <p>AI-Powered Recruitment Scam Detection using Natural Language Processing (TF-IDF) & Machine Learning (Logistic Regression)</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# Tabs for navigation
tab1, tab2, tab3 = st.tabs(["🔍 Live Detector", "📊 Model Performance", "📚 Viva & Architecture"])


# ==============================================================================
# TAB 1: LIVE DETECTOR
# ==============================================================================
with tab1:
    if not is_model_loaded:
        st.warning(
            "⚠️ The trained model and vectorizer were not found in the `models/` folder!\n\n"
            "Please train the model first by running:\n"
            "```bash\n"
            "py train.py\n"
            "```\n"
            "Once training completes, reload this page to start detecting fake jobs."
        )
    else:
        st.markdown("### 📝 Enter Job Posting Information")
        st.write("Fill in the fields below or select a sample preset to test the detection system:")

        # Preset selector
        selected_preset = st.selectbox(
            "⚡ Quick Load Sample Test Case (Great for Viva/Demo):",
            list(PRESETS.keys()),
        )

        preset_data = PRESETS[selected_preset]

        col_input1, col_input2 = st.columns([1, 1])
        with col_input1:
            job_title = st.text_input(
                "Job Title *",
                value=preset_data["title"],
                placeholder="e.g. Remote Data Entry Clerk, Senior Python Developer",
            )
        with col_input2:
            company_name = st.text_input(
                "Company Name / Profile",
                value=preset_data["company"],
                placeholder="e.g. Apex Global Solutions LLC, TechCorp Systems",
            )

        job_desc = st.text_area(
            "Job Description & Requirements *",
            value=preset_data["desc"],
            height=180,
            placeholder="Paste the full job description, qualifications, and benefits here...",
        )

        detect_button = st.button("🔍 Detect Fake Job", type="primary", use_container_width=True)

        if detect_button:
            if not job_title.strip() and not job_desc.strip():
                st.warning("⚠️ Please provide at least a Job Title or a Job Description to evaluate.")
            else:
                with st.spinner("Analyzing job posting with TF-IDF Vectorizer and ML Model..."):
                    # Combine and clean text exactly like training
                    combined_text = f"{job_title} {company_name} {job_desc}"
                    cleaned = clean_text(combined_text)

                    # Transform with TF-IDF
                    features = vectorizer.transform([cleaned])

                    # Predict
                    prediction = model.predict(features)[0]
                    probabilities = model.predict_proba(features)[0]

                    p_real = float(probabilities[0]) * 100
                    p_fake = float(probabilities[1]) * 100

                    st.markdown("---")
                    st.markdown("### 📋 Analysis Result")

                    # Heuristic keywords detection for extra explainability
                    suspicious_found = detect_suspicious_patterns(combined_text)

                    if prediction == 1:
                        # FAKE JOB
                        st.markdown(
                            f"""
                            <div class="alert-fake">
                                <h2 style="color: #991b1b; margin-top: 0;">🚨 HIGH RISK: FAKE / FRAUDULENT JOB DETECTED!</h2>
                                <p style="font-size: 1.1rem; margin-bottom: 8px;">
                                    Our machine learning model flagged this job posting as <strong>FRAUDULENT</strong> with a 
                                    <strong>{p_fake:.1f}% confidence score</strong>.
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        col_m1, col_m2 = st.columns(2)
                        with col_m1:
                            st.metric("Fraud Probability", f"{p_fake:.1f}%", delta="High Risk", delta_color="inverse")
                        with col_m2:
                            st.metric("Legitimate Probability", f"{p_real:.1f}%")

                        st.progress(p_fake / 100.0)

                        if suspicious_found:
                            st.markdown("#### 🚩 Flagged Scam Indicators & Phrases:")
                            badges_html = " ".join([f"<span class='badge badge-red'>⚠️ {kw.upper()}</span>" for kw in suspicious_found])
                            st.markdown(badges_html, unsafe_allow_html=True)

                        with st.expander("🛡️ Candidate Safety Advice & Red Flags", expanded=True):
                            st.markdown(
                                """
                                - **Never pay any upfront money** for application fees, starter kits, or background checks.
                                - **Beware of off-platform interviews** conducted solely via WhatsApp or Telegram without formal video/in-person meetings.
                                - **Unrealistic compensation:** Be cautious of promises like *$500/day for 2 hours of simple typing*.
                                - **Check official company domains:** Ensure emails come from `@company.com` and not `@gmail.com` or `@yahoo.com`.
                                """
                            )
                    else:
                        # REAL JOB
                        st.markdown(
                            f"""
                            <div class="alert-real">
                                <h2 style="color: #166534; margin-top: 0;">✅ LOW RISK: REAL / LEGITIMATE JOB POSTING</h2>
                                <p style="font-size: 1.1rem; margin-bottom: 8px;">
                                    Our machine learning model classified this job posting as <strong>LEGITIMATE</strong> with a 
                                    <strong>{p_real:.1f}% confidence score</strong>.
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        col_m1, col_m2 = st.columns(2)
                        with col_m1:
                            st.metric("Legitimate Probability", f"{p_real:.1f}%", delta="Normal", delta_color="normal")
                        with col_m2:
                            st.metric("Fraud Probability", f"{p_fake:.1f}%")

                        st.progress(p_real / 100.0)

                        with st.expander("ℹ️ Good Practices for Applying", expanded=False):
                            st.markdown(
                                """
                                - The job posting contains structured language, realistic job duties, and standard industry terminology.
                                - Always apply through the verified company career page or trusted portals (LinkedIn, Indeed).
                                - Double-check that the recruiter's identity matches the organization on professional networks.
                                """
                            )


# ==============================================================================
# TAB 2: MODEL PERFORMANCE
# ==============================================================================
with tab2:
    st.markdown("### 📈 Model Evaluation Metrics (Train / Test Split)")
    st.write(
        "Below are the actual quantitative evaluation results generated by training the "
        "Logistic Regression classifier with TF-IDF features on the test split (20% holdout)."
    )

    metrics = load_metrics()

    if metrics:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Accuracy</div>
                    <div class="metric-value">{metrics['accuracy']*100:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Precision</div>
                    <div class="metric-value">{metrics['precision']*100:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Recall</div>
                    <div class="metric-value">{metrics['recall']*100:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">F1-Score</div>
                    <div class="metric-value">{metrics['f1_score']*100:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")
        col_cm, col_info = st.columns([1, 1])

        with col_cm:
            st.markdown("#### 🔲 Confusion Matrix")
            cm = metrics["confusion_matrix"]
            cm_df = pd.DataFrame(
                cm,
                index=["Actual Real (0)", "Actual Fake (1)"],
                columns=["Predicted Real (0)", "Predicted Fake (1)"],
            )
            st.dataframe(cm_df, use_container_width=True)

            tn, fp = cm[0][0], cm[0][1]
            fn, tp = cm[1][0], cm[1][1]
            st.markdown(
                f"""
                - **True Negatives (TN):** `{tn}` (Real jobs correctly identified as real)
                - **False Positives (FP):** `{fp}` (Real jobs incorrectly flagged as fake)
                - **False Negatives (FN):** `{fn}` (Fake jobs missed by the model)
                - **True Positives (TP):** `{tp}` (Fake jobs successfully caught)
                """
            )

        with col_info:
            st.markdown("#### 📊 Dataset Breakdown")
            st.markdown(
                f"""
                - **Total Job Postings:** `{metrics['total_samples']:,}`
                - **Genuine (Real) Postings:** `{metrics['real_postings']:,}` ({metrics['real_postings']/metrics['total_samples']*100:.1f}%)
                - **Fraudulent (Fake) Postings:** `{metrics['fake_postings']:,}` ({metrics['fake_postings']/metrics['total_samples']*100:.1f}%)
                - **Training Set Size:** `{metrics['train_samples']:,}` postings (80%)
                - **Testing Set Size:** `{metrics['test_samples']:,}` postings (20%)
                """
            )
            st.info(
                "💡 **Why Recall and F1-Score Matter Most:** In fraud detection, fake jobs represent "
                "less than 5% of all listings (heavy class imbalance). A model that always guessed 'Real' "
                "would get ~95% accuracy but catch 0 fake jobs. Using balanced class weighting and measuring "
                "Precision, Recall, and F1 ensures high scam-catching capability!"
            )
    else:
        st.warning(
            "Metrics file not found! Please run `py train.py` to train the model and generate actual evaluation metrics."
        )


# ==============================================================================
# TAB 3: VIVA & ARCHITECTURE
# ==============================================================================
with tab3:
    st.markdown("### 🎓 College Project Documentation & Viva Voce Guide")

    st.markdown("#### 1. System Architecture Flow")
    st.code(
        """
Raw Job Posting (Title, Company, Description)
       ↓
Text Preprocessing (Lowercasing, Regex URL/HTML removal, Whitespace normalization)
       ↓
TF-IDF Vectorization (Extracts Unigrams & Bigrams, Computes Term Frequency - Inverse Document Frequency)
       ↓
Logistic Regression Classifier (Trained with Balanced Class Weights)
       ↓
Probability Estimation & Binary Classification (Real = 0 vs. Fake = 1)
       ↓
Streamlit Web User Interface (Interactive Form, Risk Meter, Explainable Badges)
        """,
        language="text",
    )

    st.markdown("---")
    st.markdown("#### 2. Key Machine Learning Concepts Explained")

    with st.expander("❓ What is TF-IDF and why is it used?", expanded=True):
        st.markdown(
            """
            **TF-IDF** stands for *Term Frequency - Inverse Document Frequency*.
            - **TF (Term Frequency):** Measures how frequently a word appears in a specific document.
            - **IDF (Inverse Document Frequency):** Measures how unique or rare a word is across all documents in the corpus. Common words like "the", "job", "work" receive low weights, while specific fraudulent phrases like "wire transfer", "cashapp", or "starter kit" receive high weights.
            - It converts unstructured text into numerical vectors that machine learning algorithms can compute.
            """
        )

    with st.expander("❓ Why Logistic Regression for Text Classification?"):
        st.markdown(
            """
            - **Fast and Efficient:** Works exceptionally well with high-dimensional sparse matrices produced by TF-IDF.
            - **Probabilistic Output:** Uses the sigmoid function to output calibrated probabilities (0% to 100%), allowing the web app to display a confidence score.
            - **Interpretability:** Feature weights (coefficients) can be directly inspected to see which words most strongly correlate with fraudulent postings.
            - **Class Weighting:** Supports `class_weight='balanced'` to easily counter the severe class imbalance in fraud datasets.
            """
        )

    with st.expander("❓ How does the system handle Class Imbalance?"):
        st.markdown(
            """
            In the EMSCAD dataset, fraudulent jobs make up only ~4.8% of the data. 
            We handle this using:
            1. **Stratified Train/Test Split:** Preserves the 95:5 class ratio across both training and evaluation subsets.
            2. **Balanced Class Weighting (`class_weight='balanced'`):** Inversely weights class penalties proportional to frequencies in the input data, heavily penalizing false negatives (missed scams).
            3. **Evaluation on F1-Score & Recall:** Rather than relying solely on accuracy.
            """
        )

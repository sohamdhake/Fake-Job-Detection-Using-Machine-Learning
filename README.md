# 🛡️ Fake Job Detection System
### College AIML Mini-Project | Natural Language Processing & Machine Learning

An end-to-end Machine Learning web application designed to identify fraudulent and scam job postings using **Natural Language Processing (NLP)**, **TF-IDF Vectorization**, **Logistic Regression**, and an interactive **Streamlit** dashboard.

---

## 📌 1. Project Overview & Aim
With the rise of online job recruitment portals, employment scams have surged dramatically. Fraudulent postings mislead vulnerable job seekers into identity theft, financial fraud (e.g., paying advance "training fees" or purchasing fake "starter kits"), or illegal reshipping schemes.

**Project Aim:**
To build an automated, interpretable, and accurate machine learning classification system capable of analyzing textual job postings (Title, Company Profile, Description, and Requirements) and predicting whether a job posting is **Legitimate (Real)** or **Fraudulent (Fake)**, along with a risk confidence score.

---

## 💻 2. Tech Stack & Libraries
- **Language:** Python (3.10+)
- **Data Manipulation:** `pandas`, `numpy`
- **Machine Learning & NLP:** `scikit-learn` (`TfidfVectorizer`, `LogisticRegression`, `train_test_split`, `metrics`)
- **Model Serialization:** `joblib`
- **Web Interface:** `Streamlit`

---

## 📁 3. Project Directory Structure
```text
C:\Fake-Job-Detection-System\
│
├── app.py                  # Interactive Streamlit Web Application
├── train.py                # Model training, evaluation & serialization pipeline
├── requirements.txt        # Python package dependencies
├── README.md               # Detailed project documentation & viva guide
├── fake_job_postings.csv   # Dataset (~18,000 real & fraudulent job postings)
│
└── models/                 # Saved machine learning artifacts
    ├── model.joblib        # Trained Logistic Regression classifier
    ├── vectorizer.joblib   # Fitted TF-IDF vectorizer (unigrams + bigrams)
    └── metrics.json        # Actual Accuracy, Precision, Recall, F1 evaluation results
```

---

## 📊 4. Dataset Details & Source
The system uses the benchmark **EMSCAD (Employment Scam Aegean Dataset)** / **Kaggle Real / Fake Job Posting Prediction** dataset:
- **Total Postings:** ~17,880
- **Real Postings (0):** ~17,014 (95.2%)
- **Fake Postings (1):** ~866 (4.8%)
- **Class Imbalance:** Highly skewed towards genuine jobs. This is handled using stratified splitting and `class_weight='balanced'`.

### Trusted Public Sources:
1. **Direct GitHub Mirror (Auto-downloaded by `train.py`):**
   `https://raw.githubusercontent.com/Cindyalifia/bangkit-project-1/master/fake_job_postings.csv`
2. **Kaggle Official Dataset:**
   [Kaggle: Real / Fake Job Posting Prediction by Shivamb Bansal](https://www.kaggle.com/datasets/shivamb/real-or-fake-job-posting-prediction)

> **Note:** If `fake_job_postings.csv` is not found locally, `train.py` automatically downloads it for you. You can also manually download the CSV from Kaggle and place it in the project root.

---

## 🚀 5. How to Run the Project (Step-by-Step for Windows)

### Step 1: Open Terminal in the Project Folder
Open PowerShell or Command Prompt and navigate to the project directory:
```powershell
cd C:\Fake-Job-Detection-System
```

### Step 2: Install Required Libraries
```powershell
py -m pip install -r requirements.txt
```

### Step 3: Train the Machine Learning Model
Run the training script to load data, train the TF-IDF + Logistic Regression pipeline, calculate actual evaluation metrics, and save the model:
```powershell
py train.py
```
*This will print the full classification report, confusion matrix, and save artifacts into the `models/` folder.*

### Step 4: Launch the Streamlit Web Application
```powershell
py -m streamlit run app.py
```
Your default web browser will automatically open at:
`http://localhost:8501`

---

## 📈 6. Model Evaluation & Performance Metrics
The model is evaluated using a **stratified 80/20 train/test split**:

| Metric | Description | Formula |
| :--- | :--- | :--- |
| **Accuracy** | Overall proportion of correct predictions across both classes | $\frac{TP + TN}{TP + TN + FP + FN}$ |
| **Precision** | When the model flags a posting as fake, how often it is truly fake | $\frac{TP}{TP + FP}$ |
| **Recall** | Percentage of all actual scam jobs that the model successfully caught | $\frac{TP}{TP + FN}$ |
| **F1-Score** | Harmonic mean of Precision and Recall (crucial for imbalanced fraud data) | $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ |

> **Why Balanced Class Weights?** In fraud detection, missing a scam (False Negative) is much more dangerous than asking a candidate to verify a real job (False Positive). Using `class_weight='balanced'` gives greater penalty to errors on the minority fraudulent class.

---

## 🧪 7. Sample Test Cases (For Viva / Demo)

### 🚨 Test Case 1: Fraudulent Job (Advance Fee / Work From Home Scam)
- **Job Title:** `Urgent Work From Home Data Entry Clerk`
- **Company Name:** `Apex Global Solutions LLC`
- **Description:** `We are urgently looking for remote data entry assistants! Work 1 to 2 hours per day from home and earn $500 to $800 daily. No prior experience or degree needed. Instant hiring without formal interview. To receive your company-provided laptop and training software kit, a refundable deposit of $75 must be transferred via CashApp, Venmo, or Bitcoin to our distribution agent.`
- **Expected Prediction:** 🚨 **FAKE / FRAUDULENT JOB** (~85%-98% confidence)

### ✅ Test Case 2: Legitimate Job (Software Engineer)
- **Job Title:** `Senior Python Backend Engineer`
- **Company Name:** `FinTech Innovations Corp`
- **Description:** `FinTech Innovations Corp is seeking a Senior Python Engineer with 4+ years of hands-on experience building distributed backend systems. Required skills: Python 3, FastAPI or Django, PostgreSQL, Docker, and AWS. Key responsibilities include architecting high-throughput REST APIs, writing comprehensive unit tests, and participating in code reviews. We offer competitive salary ($120k-$150k), comprehensive medical insurance, 401(k) matching, and 20 days paid vacation.`
- **Expected Prediction:** ✅ **REAL / LEGITIMATE JOB** (~95%-99% confidence)

---

## 🎓 8. College Viva Voce Questions & Answers

### Q1: What is the problem statement of this project?
> **Answer:** Online recruitment fraud harms millions of job seekers financially and emotionally. The objective is to apply Natural Language Processing and supervised Machine Learning on job descriptions and titles to automatically classify postings as authentic or fraudulent.

### Q2: What is TF-IDF and why did you choose it over Bag of Words?
> **Answer:** TF-IDF (*Term Frequency - Inverse Document Frequency*) reflects how important a word is to a document in a collection. Unlike simple Bag-of-Words (word count), TF-IDF penalizes commonly occurring stop words (like "the", "work", "job") while boosting distinct terms that signal fraud (like "cashapp", "wire transfer", "starter kit", "telegram").

### Q3: Why did you use Logistic Regression?
> **Answer:** 
> 1. Logistic Regression performs exceptionally well with sparse, high-dimensional TF-IDF feature matrices.
> 2. It outputs well-calibrated probabilities via the sigmoid function, allowing us to display confidence percentages.
> 3. It is fast, lightweight, interpretable, and computationally efficient for real-time web deployment.

### Q4: How did you handle the class imbalance in the dataset?
> **Answer:** Only ~4.8% of the dataset is fraudulent. We handled this by:
> 1. Using **Stratified Sampling** in `train_test_split` to maintain the class distribution in both splits.
> 2. Setting `class_weight='balanced'` in `LogisticRegression`, which automatically adjusts weights inversely proportional to class frequencies.
> 3. Evaluating on **Precision, Recall, and F1-Score** rather than accuracy alone.

---

## 👨‍💻 Author & Credits
- **Project:** College AIML Mini-Project
- **Dataset:** University of the Aegean (EMSCAD) / Kaggle
- **Framework:** Streamlit + Scikit-Learn

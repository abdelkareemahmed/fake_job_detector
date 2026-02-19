# 🕵️ Fake Job Posting Detector

An end-to-end Machine Learning project that detects **fraudulent job postings** using Natural Language Processing (NLP) and supervised learning.

The system analyzes job descriptions and predicts whether a posting is **real** or **fake**, along with a probability score that reflects the model’s confidence.

🔗 **Live Demo (Hugging Face Spaces):**  
👉 https://huggingface.co/spaces/abdelkareem1/fake_job_detector

---

## 🚀 Project Motivation

Fake job postings are increasingly used to scam job seekers, collect personal data, or promote misleading opportunities.  
The goal of this project is to build a **decision-aware ML system** that helps identify suspicious job postings based purely on textual content.

This project focuses not only on model performance, but also on:
- Handling **imbalanced data**
- Understanding **model behavior**
- Making **controlled, explainable decisions**

---

## 🧠 Approach Overview

The project follows a complete Machine Learning pipeline:

1. **Data Analysis**
   - Identified severe class imbalance in fraudulent vs. real job postings
2. **Text Processing**
   - Combined job title, description, and company profile into a single text field
3. **Feature Extraction**
   - Used **TF-IDF** with unigrams and bigrams
4. **Modeling**
   - Trained a **Logistic Regression** classifier as a strong, interpretable baseline
5. **Evaluation**
   - Focused on **Recall**, **Precision**, and **F1-score** instead of accuracy
6. **Decision Threshold Tuning**
   - Adjusted the probability threshold to prioritize detecting fraudulent jobs
7. **Model Behavior Analysis**
   - Precision–Recall curve
   - Probability distribution analysis
   - False positive / false negative inspection
8. **Deployment**
   - Interactive UI built with **Gradio**
   - Deployed on **Hugging Face Spaces**

---

## 📊 Model Details

- **Text Representation:** TF-IDF (unigrams + bigrams)
- **Classifier:** Logistic Regression
- **Class Imbalance Handling:** Class weighting
- **Final Decision Threshold:** `0.35` (optimized for higher recall)
- **Primary Metric:** Recall for fraudulent job postings

---

## 🧪 Example Predictions

| Input Job Posting | Fake Probability | Prediction |
|------------------|------------------|------------|
| “Work from home. No experience required. Earn quick money.” | 94.5% | 🚨 Fake |
| “Software Engineer role requiring Python and REST APIs.” | 17.1% | ✅ Real |
| “Sales Partner opportunity with flexible hours.” | 21.1% | ✅ Real |

---

## 🎯 Why Threshold Tuning Matters

Due to class imbalance, using the default probability threshold (0.5) leads to missed fraudulent postings.

Instead of optimizing for accuracy, this project:
- Prioritizes **Recall** for fraud detection
- Accepts a controlled number of false positives
- Produces **realistic and interpretable model behavior**

This makes the system suitable as a **decision-support tool**, not a black-box classifier.

---

## 🖥️ User Interface

The application provides:
- A clean text input for job descriptions
- Clear prediction labels (Real / Fake)
- A probability score indicating confidence

Built using:
- **Gradio Blocks**
- Custom layout and UX improvements
- Deployed live on Hugging Face Spaces

---

## ⚠️ Limitations

- The model relies **only on textual content**
- No verification of company identity, salary legitimacy, or URLs
- Some sophisticated scams may appear legitimate
- Intended as a **support tool**, not a definitive authority

---

## 🔮 Future Improvements

- Add metadata features (company age, domain info)
- Incorporate URL and email pattern analysis
- Experiment with transformer-based models (BERT)
- Add a “manual review” confidence zone in the UI
- Provide explanation highlights for predictions

---

## 🛠️ Tech Stack

- Python
- scikit-learn
- pandas / NumPy
- TF-IDF
- Logistic Regression
- Gradio
- Hugging Face Spaces

---

## 📁 Project Structure


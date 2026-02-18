import gradio as gr
import joblib

vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("fake_job_model.pkl")

THRESHOLD = 0.3

def predict(text):
    if not text.strip():
        return "Please enter a job description."

    vec = vectorizer.transform([text])
    prob = model.predict_proba(vec)[0][1]

    label = "🚨 Fake Job" if prob >= THRESHOLD else "✅ Real Job"

    return f"{label}\nFake Probability: {prob:.2%}"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        lines=8,
        placeholder="Paste job description here..."
    ),
    outputs=gr.Textbox(),
    title="Fake Job Posting Detector",
    description="Machine Learning model that detects fake job postings."
)

demo.launch()

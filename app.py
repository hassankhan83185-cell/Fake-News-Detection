from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    news_text = request.form['news']

    # Convert text into vector
    news_vector = vectorizer.transform([news_text])

    # Prediction
    prediction = model.predict(news_vector)[0]

    # Confidence score
    confidence = model.predict_proba(news_vector).max() * 100

    if prediction == 1:
        result = "REAL NEWS ✅"
        color = "#22c55e"
        message = "This news appears to be authentic according to the AI model."
    else:
        result = "FAKE NEWS ❌"
        color = "#ef4444"
        message = "This news appears suspicious and may contain misinformation."

    return f"""
    <style>

    body{{
        background:#0f172a;
        color:white;
        font-family:Arial, sans-serif;
        display:flex;
        justify-content:center;
        align-items:center;
        height:100vh;
        margin:0;
    }}

    .result-card{{
        background:#111827;
        padding:40px;
        border-radius:20px;
        text-align:center;
        width:700px;
        box-shadow:0 0 30px rgba(0,245,255,0.2);
    }}

    h1{{
        color:#00f5ff;
    }}

    h2{{
        color:{color};
    }}

    .confidence{{
        color:#00f5ff;
        font-size:22px;
        font-weight:bold;
    }}

    .news-box{{
        margin-top:20px;
        background:#1f2937;
        padding:20px;
        border-radius:10px;
        text-align:left;
    }}

    a{{
        display:inline-block;
        margin-top:30px;
        padding:12px 25px;
        background:#00f5ff;
        color:black;
        text-decoration:none;
        border-radius:10px;
        font-weight:bold;
    }}

    </style>

    <div class="result-card">

        <h1>Truth Lens Analysis 🚀</h1>

        <h2>{result}</h2>

        <p class="confidence">
            Confidence Score: {confidence:.2f}%
        </p>

        <p>{message}</p>

        <div class="news-box">
            <strong>News Text:</strong><br><br>
            {news_text}
        </div>

        <a href="/">Analyze Another News</a>

    </div>
    """


if __name__ == '__main__':
    app.run(debug=True)
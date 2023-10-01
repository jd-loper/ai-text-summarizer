from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/summarize", methods=["POST"])
def summarize_text():
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    text = request.form["text"]
    max_length = min(50, len(text.split()) // 2)
    summary = summarizer(text, max_length=max_length, min_length=25, do_sample=False)
    return render_template("result.html", summary=summary[0]["summary_text"])


if __name__ == "__main__":
    app.run(debug=True)

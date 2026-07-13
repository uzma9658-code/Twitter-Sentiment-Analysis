from flask import Flask, render_template, request
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load model
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Load dataset
data = pd.read_excel("twitter_sentiment.csv.xlsx")

countries = sorted(data["Country"].dropna().unique())


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyzer", methods=["GET", "POST"])
def analyzer():
    prediction = None
    emoji = None

    if request.method == "POST":
        text = request.form["text"]
        vectorized = vectorizer.transform([text])
        result = model.predict(vectorized)[0].lower()

        if result == "positive":
            prediction = "Positive"
            emoji = "😊✨🔥"
        elif result == "negative":
            prediction = "Negative"
            emoji = "😡💔"
        else:
            prediction = "Neutral"
            emoji = "😐"

    return render_template("analyzer.html",
                           prediction=prediction,
                           emoji=emoji)


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    selected_country = None
    chart_path = None

    if request.method == "POST":
        selected_country = request.form["country"]
        country_data = data[data["Country"] == selected_country]

        sentiment_counts = country_data["sentiment"].value_counts()

        # Create chart
        plt.figure()
        sentiment_counts.plot(kind="bar")
        plt.title(f"Sentiment Analysis - {selected_country}")
        plt.xlabel("Sentiment")
        plt.ylabel("Count")

        chart_path = f"static/charts/{selected_country}.png"
        plt.savefig(chart_path)
        plt.close()

    return render_template("dashboard.html",
                           countries=countries,
                           selected_country=selected_country,
                           chart_path=chart_path)


if __name__ == "__main__":
    app.run(debug=True)

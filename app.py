from flask import Flask, render_template, request, redirect, send_file
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd
import csv
import os

app = Flask(__name__)
API_KEY = "AIzaSyD4pMAKWDzb5qAu2L4anvwysavnTqJ7GRk"

def build_youtube_client(api_key):
    return build("youtube", "v3", developerKey=api_key)

def get_comments(client, video_id, token=None):
    try:
        response = client.commentThreads().list(
            part="snippet",
            videoId=video_id,
            textFormat="plainText",
            maxResults=100,
            pageToken=token,
        ).execute()
        return response
    except HttpError as e:
        print(f"HTTP Error: {e.resp.status}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def fetch_all_comments(client, video_id):
    comments = []
    next_token = None
    while True:
        response = get_comments(client, video_id, next_token)
        if not response:
            break
        comments.extend(response.get("items", []))
        next_token = response.get("nextPageToken")
        if not next_token:
            break
    return comments

def save_comments_to_csv(comments, output_file):
    with open(output_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Comment"])
        for item in comments:
            text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            writer.writerow([text])

def analyze_sentiment(input_csv, output_csv):
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    df = pd.read_csv(input_csv)
    df["Sentiment"] = df["Comment"].astype(str).apply(lambda text: classifier(text[:512])[0]["label"])
    df.to_csv(output_csv, index=False)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    video_id = request.form["video_id"]
    output_file = request.form["output_file"]
    analyze_flag = "analyze" in request.form

    yt_client = build_youtube_client(API_KEY)
    comments = fetch_all_comments(yt_client, video_id)
    save_comments_to_csv(comments, output_file)

    if analyze_flag:
        analyzed_file = output_file.replace(".csv", "_analyzed.csv")
        analyze_sentiment(output_file, analyzed_file)
        return send_file(analyzed_file, as_attachment=True)
    else:
        return send_file(output_file, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
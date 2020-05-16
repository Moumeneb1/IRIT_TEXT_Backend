from flask import Flask, jsonify, request
import uuid
from scrapper import Scrapper
import datetime
import sys

app = Flask(__name__)
scraper = Scrapper()
sessions = dict()


@app.route('/', methods=['GET', 'POST'])
def index():
    print("hello worlddd")
    if (request.method == "POST"):
        some_json = request.json()
        return jsonify({
            "you sent": some_json,
        })
    else:
        return jsonify({"about": "hello_world"})


@app.route('/tweet_data/', methods=['POST'])
def scrap(ID):
    if request.method == 'POST':
        ID = request.json['ID']
        tweet = scraper.get_tweet_byID(ID)
        return jsonify(tweet)


@app.route('/scrap_tweets/', methods=['POST'])
def scrap_df():
    if request.method == 'POST':
        # we will get the file from the request
        #keywords = request.form.getlist('keywords')
        data = request.json
        session_token = str(uuid.uuid4())
        lang = data["lang"]
        limit = data["limit"]
        begin_date = datetime.datetime.strptime(data["begin_date"],
                                                "%Y-%m-%d").date()
        end_date = datetime.datetime.strptime(data["end_date"],
                                              "%Y-%m-%d").date()
        keywords = data["keywords"]
        keywords = [str(r) for r in keywords]  #Remove encoding
        df = scraper.get_tweets_df(keywords=keywords,
                                   lang=lang,
                                   begindate=begin_date,
                                   enddate=end_date,
                                   limit=10)

        sessions[session_token] = df
        print(sessions.keys())
        return jsonify({
            'session_token': session_token,
            'dataframe': df.to_json(orient="records")
        })


@app.route("/predict_dataframe", methods=["POST"])
def predict():
    if request.method == 'POST':
        data = request.json
        session_token = data["session_token"]
        session_token = session_token.encode("ascii", "replace")
        print(sessions.keys())
        print(len(sessions))
        df = sessions[session_token]
        #Feature enginnering
        #Preprocessing
        #Text
        return jsonify({'session_token': session_token, 'test': 1})


if __name__ == "__main__":
    app.run(debug=True)
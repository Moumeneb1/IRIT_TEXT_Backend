from flask import Flask, jsonify , request 
from Scrapper import Scrapper
import datetime
import sys
 

app = Flask(__name__)
scraper = Scrapper()

@app.route('/',methods=['GET','POST'] )
def index():
    print("hello worlddd")
    if (request.method =="POST"):
        some_json = request.json()
        return jsonify({
            "you sent": some_json,
        })
    else:
        return jsonify({
            "about" : "hello_world"
        })


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
        lang = data["lang"]
        limit = data["limit"]
        begin_date = datetime.datetime.strptime(data["begin_date"], "%Y-%m-%d").date()
        end_date = datetime.datetime.strptime(data["end_date"], "%Y-%m-%d").date()
        keywords = data["keywords"]
        keywords = [str(r) for r in keywords]#Remove encoding 
        df=scraper.get_tweets_df(keywords=keywords ,lang=lang,begindate=begin_date, enddate=end_date,limit=10)
        return jsonify(df.to_json(orient="records"))


if __name__ =="__main__":
    app.run(debug=True)
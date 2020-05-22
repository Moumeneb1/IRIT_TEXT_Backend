from flask import Flask, jsonify, request
import uuid
from scrapper import Scrapper
import datetime
import sys
from preprocessing.text_preprocessing import TextPreprocessing
from preprocessing.feature_enginering import FeaturesExtraction
from model_dict import get_model
from bertInput import BertInput
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np


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
        lang = 'fr'
        limit = data["limit"]
        begin_date = datetime.datetime.strptime(data["begin_date"],
                                                "%m-%d-%Y").date()
        end_date = datetime.datetime.strptime(data["end_date"],
                                              "%m-%d-%Y").date()
        keywords = data["keywords"]
        keywords = [str(r) for r in keywords]  # Remove encoding
        df = scraper.get_tweets_df(keywords=keywords,
                                   lang=lang,
                                   begindate=begin_date,
                                   enddate=end_date,
                                   limit=10)

        sessions[session_token] = df
        print(sessions.keys())
        return jsonify({
            'session_token': session_token,
            'dataframe_length': df.shape[0]
        })


@app.route("/predict_dataframe", methods=["POST"])
def predict():
    if request.method == 'POST':
        data = request.json
        session_token = data["session_token"]
        model_name = data["model_name"]
        domain_name = data["domain"]
        session_token = session_token.encode("ascii", "replace")
        df = sessions[session_token]
        # Feature enginnering

        featuresExtrator = FeaturesExtraction(df, "texte")
        featuresExtrator.fit_transform()

        # Preprocessing
        text_preprocessing = TextPreprocessing(df, "texte")
        text_preprocessing.fit_transform()

        # Load model ,Tokenizer , labels_dict , features

        model, tokenizer, labels_dict, features = get_model(
            domain_name, model_name)

        # get text
        sentences = df["texte"]
        bert_input = BertInput(tokenizer)
        sentences = bert_input.fit_transform(sentences)
        input_ID = torch.tensor(sentences[0])
        input_MASK = torch.tensor(sentences[1])
        print(len(sentences))
        if features:
            features_column = df[features].values.astype(float).tolist()
            features_column = torch.tensor(features_column)
            tensor_dataset = TensorDataset(
                input_ID, input_MASK, features_column)
        else:
            tensor_dataset = TensorDataset(sentences)

        dataloader = DataLoader(
            tensor_dataset, batch_size=1, shuffle=False, num_workers=4)

        pred = []
        for index, batch in enumerate(dataloader):
            output = model(batch)
            label_index = np.argmax(output[0].cpu().detach().numpy())
            pred.append(labels_dict.get(label_index))
        df['prediction'] = pred

        # Inference
        return jsonify({
            'session_token': session_token,
            'dataframe': df.to_json(orient="records")
        })


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=4000)

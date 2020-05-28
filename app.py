from flask import Flask, jsonify, request, Response
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
from flask_cors import CORS, cross_origin
from model_dict import models_dic
import pandas as pd


app = Flask(__name__)
cors = CORS(app)

scraper = Scrapper()
sessions = dict()


@app.route('/api/get_noFeatures_models/', methods=['GET'])
@cross_origin()
def get_noFeatures_models():

    dict_ = []
    for i, k in enumerate(models_dic):
        dict_.append({})
        dict_[i]['name'] = k
        dict_[i]['models'] = []
        j = 0
        for k_, v_ in models_dic[k]['models'].items():
            if 'features' not in models_dic[k]['models'][k_]:
                dict_[i]['models'].append({'name': k_})

    response = jsonify(dict_)
    return response


@app.route('/api/get_all_models/', methods=['GET'])
@cross_origin()
def get_all_models():

    dict_ = []
    for i, k in enumerate(models_dic):
        dict_.append({})
        dict_[i]['name'] = k
        dict_[i]['models'] = [{'name': v_} for k_,
                              v_ in enumerate(models_dic[k]['models'])]

    response = jsonify(dict_)
    return response


@app.route('/api/tweet_data/', methods=['POST'])
@cross_origin()
def scrap(ID):
    if request.method == 'POST':
        ID = request.json['ID']
        tweet = scraper.get_tweet_byID(ID)
        return jsonify(tweet)


@app.route('/api/scrap_tweets/', methods=['POST'])
@cross_origin()
def scrap_df():
    if request.method == 'POST':
        # we will get the file from the request
        #keywords = request.form.getlist('keywords')
        print(request.json)
        data = request.json
        session_token = str(uuid.uuid4())
        lang = 'fr'
        limit = int(data["limit_scrap"])
        begin_date = datetime.datetime.strptime(data["begin_date"],
                                                "%m/%d/%Y").date()
        end_date = datetime.datetime.strptime(data["end_date"],
                                              "%m/%d/%Y").date()
        keywords = data["keywords"]
        keywords = [str(r) for r in keywords]  # Remove encoding
        df = scraper.get_tweets_df(keywords=keywords,
                                   lang=lang,
                                   begindate=begin_date,
                                   enddate=end_date,
                                   limit=limit)
        df.drop_duplicates(subset='id', keep="last")
        sessions[session_token] = df
        print(sessions.keys())
        response = jsonify({
            'session_token': session_token,
            'dataframe_length': df.shape[0]
        })
        return response


@app.route("/api/predict_dataframe", methods=["POST"])
@cross_origin()
def predict():
    if request.method == 'POST':
        data = request.json
        print(sessions)
        session_token = str(data["session_token"])
        model_name = str(data["model_name"])
        domain_name = str(data["field"])
        session_token = session_token
        print(session_token)
        df = sessions[session_token]
        # Feature enginnering

        print(df)
        featuresExtrator = FeaturesExtraction(df, "text")
        featuresExtrator.fit_transform()

        # Preprocessing
        text_preprocessing = TextPreprocessing(df, "text")
        text_preprocessing.fit_transform()

        # drop small-text columns
        df = df[~(df['processed_text'].str.len() > 100)]
        #df = df[len(df['processed_text']) > 60]
        # Load model ,Tokenizer , labels_dict , features

        model, tokenizer, labels_dict, features = get_model(
            domain_name, model_name)

        # get text
        sentences = df["text"]
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
            tensor_dataset = TensorDataset(input_ID, input_MASK)
        dataloader = DataLoader(
            tensor_dataset, batch_size=1, shuffle=False, num_workers=4)

        pred = []
        for index, batch in enumerate(dataloader):
            output = model(batch)
            label_index = np.argmax(output[0].cpu().detach().numpy())
            print(index)
            pred.append(labels_dict.get(label_index))
        df['prediction'] = pred

        # Inference
        response = jsonify({
            'session_token': session_token,
            'dataframe': df.to_json(orient="records"),
            'summary': df['prediction'].value_counts().to_json(),
        })

        return response


@app.route("/api/predict_onetweet", methods=["POST"])
@cross_origin()
def predict_one():
    if request.method == 'POST':
        data = request.json
        print(data)
        model_name = str(data["model_name"])
        domain_name = str(data["field"])
        df = pd.DataFrame.from_dict({"text": [data['text']]})
        print(data['text'])

        # Feature enginnering
        print(df)
        featuresExtrator = FeaturesExtraction(df, "text")
        featuresExtrator.fit_transform()

        # Preprocessing
        text_preprocessing = TextPreprocessing(df, "text")
        text_preprocessing.fit_transform()

        # drop small-text columns
        df = df[~(df['processed_text'].str.len() > 100)]
        #df = df[len(df['processed_text']) > 60]
        # Load model ,Tokenizer , labels_dict , features

        model, tokenizer, labels_dict, features = get_model(
            domain_name, model_name)

        # get text
        sentences = df["text"]
        bert_input = BertInput(tokenizer)
        sentences = bert_input.fit_transform(sentences)
        input_ID = torch.tensor(sentences[0])
        input_MASK = torch.tensor(sentences[1])
        print(len(sentences))

        pred = []
        output = model((input_ID, input_MASK,))
        label_index = np.argmax(output[0].cpu().detach().numpy())
        pred.append(labels_dict.get(label_index))
        df['prediction'] = pred

        print(df['prediction'].iloc[0])
        # Inference
        response = jsonify({
            'prediction': df['prediction'].iloc[0],
        })

        return response


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=4000)

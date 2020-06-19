from models import BasicBertForClassification, BertFeaturesForSequenceClassification
from transformers import AutoTokenizer

# When you add a model or a domain to your app Just import your model and add the path to it
models_dic = {
    "Crisis":{
    "crisis_binary": {
        'models': {
            "flaubert-base-cased": {
                "model": BasicBertForClassification,
                "path": "../models_weights/Crisis Binary/Crisis_Binary_flaubert_base.pth",
                "tokenizer_base": "flaubert-base-cased",
            },
        },
        "labels_dic": {
            0: 'Message-Utilisable',
            1: 'Message-NonUtilisable'
        }
    },
    "crisis_Three_Class": {
        'models': {
            "flaubert-base-cased": {
                "model": BasicBertForClassification,
                "path": "../models_weights/Crisis Three Classes/Crisis_ThreeClass_flaubert_base.pth",
                "tokenizer_base": "flaubert-base-cased",
            },
        },
        "labels_dic": {
            0: 'Message-InfoUrgent',
            1: 'Message-NonUtilisable',
            2: 'Message-InfoNonUrgent', }
    },
    "crisis_MultiClass": {
        'models': {
            "bert_base_multiligual_cased": {
                "model": BasicBertForClassification,
                "path": "../models_weights/Crisis MultiClass/Crisis_MultiClass_flaubert_base.pth",
                "tokenizer_base": "flaubert-base-cased",
            },

        },
        "labels_dic": {
            0: 'Degats-Materiels',
            1: 'Degats-Humains',
            2: 'AutresMessages',
            3: 'Message-NonUtilisable',
            4: 'Avertissement-conseil',
            5: 'Soutiens',
            6: 'Critiques'}
    }},
    "Psycho":{
    "psycho_sentiment": {
        'models': {
            "flaubert_base_cased": {
                "model": BasicBertForClassification,
                "path": "../models_weights/Psycho sentiment/Psycho_sentiment_flaubert_base.pth",
                "tokenizer_base": "flaubert-base-cased",
            },
            "bert_adapted": {
                "model": BasicBertForClassification,
                "path": "../models_weights/Psycho sentiment/Pycho_sentiment_bert_adepted.pth",
                "tokenizer_base": "bert-base-multilingual-cased",
            },
            "bert_base_cased": {
                "model": BasicBertForClassification,
                "path": "../models_weights/Psycho sentiment/Pycho_sentiment_bert_base.pth",
                "tokenizer_base": "bert-base-multilingual-cased",
            },
            "flaubert_adapted": {
                "model": BasicBertForClassification,
                "path": "../models_weights/Psycho sentiment/Pycho_sentiment_flaubert_adapted.pth",
                "tokenizer_base": "flaubert-base-cased",
            },
            "flaubert_base": {
                "model": BasicBertForClassification,
                "path": "../models_weights/Psycho sentiment/Pycho_sentiment_flaubert_base.pth",
                "tokenizer_base": "flaubert-base-cased",
            },

        },
        "labels_dic": {
            0: 'opinionPositive',
            1: 'opinionNegative',
            2: 'sansOpinion-ou-mixte'},

    },
    "psycho_useCase": {
        'models': {
            "bert_base_cased": {
                "model": BasicBertForClassification,
                "path": "../models_weights/Psycho useCase/Pycho_useCase_bert_base.pth",
                "tokenizer_base": "bert-base-multilingual-cased",
            },
            "bert_adapted": {
                "model": BasicBertForClassification,
                "path": "../models_weights/Psycho useCase/Pycho_useCase_bert_adapted.pth",
                "tokenizer_base": "bert-base-multilingual-cased",
            },
            "flaubert_base": {
                "model": BasicBertForClassification,
                "path": "../models_weights/Psycho useCase/Pycho_useCase_flaubert_adapted.pth",
                "tokenizer_base": "flaubert-base-cased",
            },
            "flaubert_adapted": {
                "model": BasicBertForClassification,
                "path": "../models_weights/Psycho sentiment/Pycho_sentiment_flaubert_adapted.pth",
                "tokenizer_base": "flaubert-base-cased",
            },
            "flaubert_base": {
                "model": BasicBertForClassification,
                "path": "../models_weights/Psycho useCase/Pycho_useCase_flaubert_base.pth",
                "tokenizer_base": "flaubert-base-cased",
            },

        },

        "labels_dic": {
            0: 'UsageDetourne',
            1: 'UsageMedical',
            2: 'Poubelle'}
    }},
}


def get_model(domain, model_name):
    


    models = {}
    for d in list(models_dic.values()):
       models.update(d)
    model = models[domain]['models'][model_name]["model"].load(
        models[domain]['models'][model_name]["path"])

    if "features" in models[domain]['models'][model_name]:
        features = models[domain]['models'][model_name]["features"]
    else:
        features = []
    Tokenizer = AutoTokenizer.from_pretrained(
        models[domain]['models'][model_name]["tokenizer_base"])

    return model, Tokenizer, models[domain]["labels_dic"], features

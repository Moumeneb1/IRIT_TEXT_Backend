from models import BasicBertForClassification, BertFeaturesForSequenceClassification
from transformers import AutoTokenizer

# When you add a model or a domain to your app Just import your model and add the path to it
models_dic = {
    "crisis_binary": {
        "bert_base_cased": {
            "model": BasicBertForClassification,
            "path": "dqdsqdq",
        },
        "flaubert_base_cased": {
            "model": BasicBertForClassification,
            "path": "dqshdbjq"
        },
        "flaubert_base_features": {
            "model": BertFeaturesForSequenceClassification,
            "path": "dqshdbjq",
            "features": ['nbretweet', 'nblike']
        },
        "labels_dic": {
            0: 'Message-Utilisable',
            1: 'Message-NonUtilisable'
        }
    },
    "crisis_Three_Class": {
        "bert_base_cased": {
            "model": BasicBertForClassification,
            "path": "dqdsqdq"
        },
        "flaubert_base_cased": {
            "model": BasicBertForClassification,
            "path": "dqshdbjq"
        },
        "flaubert_base_features": {
            "model": BasicBertForClassification,
            "path": "dqshdbjq",
            "fatures": ["dqsdqsd", ""]
        },
        "labels_dic": {
            0: 'Message-InfoUrgent',
            1: 'Message-InfoNonUrgent',
            2: 'Message-NonUtilisable'}
    },
    "crisis_MultiClass": {
        "bert_base_cased": {
            "model": BasicBertForClassification,
            "path": "dqdsqdq"
        },
        "flaubert_base_cased": {
            "model": BasicBertForClassification,
            "path": "dqshdbjq"
        },
        "flaubert_base_features": {
            "model": BasicBertForClassification,
            "path": "dqshdbjq",
            "fatures": ["dqsdqsd", ""]
        },
        "labels_dic": {
            0: 'Degats-Materiels',
            1: 'Avertissement-conseil',
            2: 'AutresMessages',
            3: 'Message-NonUtilisable',
            4: 'Soutiens',
            5: 'Degats-Humains',
            6: 'Critiques'}
    },
    "psycho_sentiment": {
        "bert_base_cased": {
            "model": BasicBertForClassification,
            "path": "dqdsqdq"
        },
        "flaubert_adapted_features": {
            "model": BertFeaturesForSequenceClassification,
            "path": "flaubert_classification.pth",
            "tokenizer_base": "flaubert-base-cased",
            "features": ['nbretweet', 'nblike'],
        },
        "labels_dic": {
            0: 'opinionNegative',
            1: 'sansOpinion-ou-mixte',
            2: 'opinionPositive'}
    },
    "psycho_use": {
        "model": BasicBertForClassification,
        "path": ",qldks,qdl"
    }
}


def get_model(domain, model_name):
    model = models_dic[domain][model_name]["model"].load(
        models_dic[domain][model_name]["path"])

    if "features" in models_dic[domain][model_name]:
        features = models_dic[domain][model_name]["features"]
    else:
        features = []
    Tokenizer = AutoTokenizer.from_pretrained(
        models_dic[domain][model_name]["tokenizer_base"])

    return model, Tokenizer, models_dic[domain]["labels_dic"], features

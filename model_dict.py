from models import BasicBertForClassification, BertFeaturesForSequenceClassification
from transformers import AutoTokenizer

# When you add a model or a domain to your app Just import your model and add the path to it
models_dic = {
    "crisis_binary": {
        'models': {
            "flaubert-base-cased": {
                "model": BasicBertForClassification,
                "path": "../PFE/nlpcrisis/Codes/deep_learning/my_models/models_weights/Crisis Binary/Crisis_Binary_flaubert_base.pth",
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
                "path": "../PFE/nlpcrisis/Codes/deep_learning/my_models/models_weights/Crisis Three Classes/Crisis_ThreeClass_flaubert_base.pth",
                "tokenizer_base": "flaubert-base-cased",
            },
        },
        "labels_dic": {
            0: 'Message-InfoUrgent',
            1: 'Message-InfoNonUrgent',
            2: 'Message-NonUtilisable'}
    },
    "crisis_MultiClass": {
        'models': {
            "bert_base_multiligual_cased": {
                "model": BasicBertForClassification,
                "path": "../PFE/nlpcrisis/Codes/deep_learning/my_models/models_weights/Crisis MultiClass/Crisis_MultiClass_flaubert_base.pth",
                "tokenizer_base": "flaubert-base-cased",
            },

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
        'models': {
            "flaubert_base_cased": {
                "model": BasicBertForClassification,
                "path": "../PFE/nlpcrisis/Codes/deep_learning/my_models/models_weights/Psycho sentiment/Psycho_sentiment_flaubert_base.pth",
                "tokenizer_base": "flaubert-base-cased",
            },
            "bert_adapted": {
                "model": BasicBertForClassification,
                "path": "../PFE/nlpcrisis/Codes/deep_learning/my_models/models_weights/Psycho sentiment/Pycho_sentiment_bert_adepted.pth",
                "tokenizer_base": "bert-base-multilingual-cased",
            },
            "bert_base_cased": {
                "model": BasicBertForClassification,
                "path": "../PFE/nlpcrisis/Codes/deep_learning/my_models/models_weights/Psycho sentiment/Pycho_sentiment_bert_base.pth",
                "tokenizer_base": "bert-base-multilingual-cased",
            },
            "flaubert_adapted": {
                "model": BasicBertForClassification,
                "path": "../PFE/nlpcrisis/Codes/deep_learning/my_models/models_weights/Psycho sentiment/Pycho_sentiment_flaubert_adapted.pth",
                "tokenizer_base": "flaubert-base-cased",
            },
            "flaubert_base": {
                "model": BasicBertForClassification,
                "path": "../PFE/nlpcrisis/Codes/deep_learning/my_models/models_weights/Psycho sentiment/Pycho_sentiment_flaubert_base.pth",
                "tokenizer_base": "flaubert-base-cased",
            },

        },
        "labels_dic": {
            0: 'opinionNegative',
            1: 'sansOpinion-ou-mixte',
            2: 'opinionPositive'},

    },
    "psycho_useCase": {
        'models': {
            "bert_base_cased": {
                "model": BasicBertForClassification,
                "path": "../PFE/nlpcrisis/Codes/deep_learning/my_models/models_weights/Psycho useCase/Pycho_useCase_bert_base.pth",
                "tokenizer_base": "bert-base-multilingual-cased",
            },
            "bert_adapted": {
                "model": BasicBertForClassification,
                "path": "../PFE/nlpcrisis/Codes/deep_learning/my_models/models_weights/Psycho useCase/Pycho_useCase_bert_adapted.pth",
                "tokenizer_base": "bert-base-multilingual-cased",
            },
            "flaubert_base": {
                "model": BasicBertForClassification,
                "path": "../PFE/nlpcrisis/Codes/deep_learning/my_models/models_weights/Psycho useCase/Pycho_useCase_flaubert_adapted.pth",
                "tokenizer_base": "flaubert-base-cased",
            },
            "flaubert_adapted": {
                "model": BasicBertForClassification,
                "path": "../PFE/nlpcrisis/Codes/deep_learning/my_models/models_weights/Psycho sentiment/Pycho_sentiment_flaubert_adapted.pth",
                "tokenizer_base": "flaubert-base-cased",
            },
            "flaubert_base": {
                "model": BasicBertForClassification,
                "path": "../PFE/nlpcrisis/Codes/deep_learning/my_models/mode/home/aboumada/PFE/nlpcrisis/Codes/deep_learning/my_models/models_weights/Psycho useCase/Pycho_useCase_flaubert_base.pth",
                "tokenizer_base": "flaubert-base-cased",
            },

        },

        "labels_dic": {
            0: 'opinionNegative',
            1: 'sansOpinion-ou-mixte',
            2: 'opinionPositive'}
    },
}


def get_model(domain, model_name):
    model = models_dic[domain]['models'][model_name]["model"].load(
        models_dic[domain]['models'][model_name]["path"])

    if "features" in models_dic[domain]['models'][model_name]:
        features = models_dic[domain]['models'][model_name]["features"]
    else:
        features = []
    Tokenizer = AutoTokenizer.from_pretrained(
        models_dic[domain]['models'][model_name]["tokenizer_base"])

    return model, Tokenizer, models_dic[domain]["labels_dic"], features

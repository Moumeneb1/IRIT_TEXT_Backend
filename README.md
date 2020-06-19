# Easy NLP Backend

## How To use

Clone this Repo

```bash
$ git clone https://github.com/Moumeneb1/IRIT_TEXT_Backend.git
$ cd IRIT_TEXT_Backend
$ pip install -r requirements.txt
```

To make the backend flexibale to new domains and models we use a model_dict, dictionnary on [**model_dict.py**](./model_dict.py)

```python
>>> models_dic = {
>>>     "<Your domain>":{
>>>            "crisis_binary": {
>>>                'models': {
>>>                    "flaubert-base-cased": {
>>>                        "model": BasicBertForClassification, #model
>>>                        "path": "../models_weights/Crisis Binary/Crisis_Binary_flaubert_base.pth"#path
>>>                        "tokenizer_base": "flaubert-base-cased",#base tokenizer
>>>                },
>>>            },
>>>                "labels_dic": { #To not load pickle objects we prefered filling them manually
>>>                    0: 'Message-Utilisable',
>>>                    1: 'Message-NonUtilisable'
>>>                }
>>>           },
>>>    }
}

```

3. Run the Server

```bash
    python app.py
```

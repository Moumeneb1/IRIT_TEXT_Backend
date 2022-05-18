from unittest import result
from flask import Flask, request, jsonify, abort,make_response
from index import Indexer
import flasgger
from flasgger import Swagger
import json
from dotenv import load_dotenv
import os
from flask_cors import CORS, cross_origin


app = Flask(__name__)
cors = CORS(app)

Swagger(app)

if not load_dotenv():
    print("env file not loaded correctly")

regenerate_index = True if os.getenv('REGENERATE_INDEX')=="TRUE" else False

index = Indexer(regenerate_index=regenerate_index)


@app.route('/semantic_search', methods=['GET'])
@cross_origin()
def semantic_search():
    # handle the POST request
    """Let's do a semantic search
    Let's do a semantic search.
    ---
    parameters:
      - name: query
        in: query
        type: string
        required: true
      - name: file_type
        in: query
        type: string
        enum: ['wavs', 'documents', 'all']
        required: true
      - name: top_k
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
    """
    if request.method == 'GET':
        query = request.args.get('query')
        file_type = request.args.get('file_type')
        top_k = int(request.args.get('top_k'))
    try:
        data = index.search(query,top_k,file_type)
        response  = make_response(jsonify(data), 201)
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response
    except Exception as e:
         abort(500)

@app.errorhandler(500)
def internal_error(error):
  return ("500 error - Internal Server Exception",500)

if __name__== '__main__':
    app.run(debug=True, host='0.0.0.0',port=80)

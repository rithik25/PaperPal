from flask import Flask, request, jsonify
from flask_cors import CORS
from paper_ranker import query_chatgpt
import pickle

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains and routes

with open('hits_scores.pickle', 'rb') as f:
    hits_scores = pickle.load(f)

with open('bm25_model.pickle', 'rb') as f:
    bm25_model = pickle.load(f)

with open('doc_texts.pickle', 'rb') as f:
    doc_texts = pickle.load(f)

with open('doc_authors.pickle', 'rb') as f:
    doc_authors = pickle.load(f)

with open('doc_titles.pickle', 'rb') as f:
    doc_titles = pickle.load(f)

@app.route('/submit_query', methods=['POST'])
def submit_query():
    data = request.json
    user_input = data['userInput']
    response = query_chatgpt(user_input, hits_scores, bm25_model, doc_texts, doc_authors, doc_titles)
    #print(user_input)
    return jsonify({'response': response})
    #jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
